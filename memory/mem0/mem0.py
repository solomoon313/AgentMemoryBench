from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..base import MemoryMechanism
from src.utils.message_schema import (
    extract_message_info,
    enhance_messages_with_memory,
    extract_original_question,
)


def _serialize_history(history: List[Any], template_title: str, where: str) -> List[Dict[str, Any]]:
    """
    Convert history to a serializable format (JSON compatible).
    Filters out non-chat messages such as RewardHistoryItem and converts Pydantic models to dicts.
    Key: must strip memory content injected in this round, keeping only the original interaction.

    Args:
        history: Conversation history
        template_title: Template title (used to identify injected memory content)
        where: Insertion position ("tail" or "front")
    """
    template_titles = [template_title]
    serialized = []

    for msg in history:
        role, content, msg_dict = extract_message_info(msg)

        # Skip messages where role cannot be extracted (e.g. RewardHistoryItem)
        if role is None:
            continue

        # If this is a user message containing injected memory, extract the original question
        if role == "user" and content:
            # Check if the message contains injected memory
            from src.utils.message_schema import ORIGINAL_QUESTION_SEPARATOR
            has_memory = (
                ORIGINAL_QUESTION_SEPARATOR in str(content) or
                any(title in str(content) for title in template_titles)
            )

            if has_memory:
                # Extract the original question
                question = extract_original_question([msg], where=where, template_titles=template_titles)
                if question:
                    content = question

        # Use the full message dict if available
        if msg_dict is not None:
            # Ensure it is a dict
            if isinstance(msg_dict, dict):
                # Create a new dict with the filtered content
                filtered_msg = dict(msg_dict)
                filtered_msg["content"] = str(content) if content else ""
                serialized.append(filtered_msg)
            else:
                # Other types: create a minimal dict
                serialized.append({
                    "role": role,
                    "content": str(content) if content else ""
                })
        else:
            # No full dict available: create a minimal dict
            serialized.append({
                "role": role,
                "content": str(content) if content else ""
            })

    return serialized

# Import Mem0 Platform client
try:
    from mem0 import MemoryClient
    HAS_MEM0 = True
except ImportError:
    HAS_MEM0 = False


@dataclass
class Mem0Config:
    api_key: str = ""
    user_id: str = "default"  # User-defined user_id (no longer fixed to task)
    infer: bool = True
    top_k: int = 5
    threshold: Optional[float] = 0.7
    rerank: bool = True
    success_only: bool = True
    reward_bigger_than_zero: bool = False  # True: only store samples with reward > 0, False: store all
    prompt_template: str = "Based on your previous interactions, here are relevant memories:\n{memories}"
    where: str = "tail"  # "tail": memory appended after user question | "front": memory prepended before user question
    # Retry configuration
    max_retries: int = -1  # -1: unlimited retries, 0: no retry, >0: max retry count
    retry_delay: float = 1.0  # Retry delay (seconds), initial value for exponential backoff
    retry_backoff: float = 2.0  # Exponential backoff multiplier
    # Wait configuration
    wait_time: float = 0.0  # Time to wait after each successful add (seconds), to avoid request bursts


class Mem0Memory(MemoryMechanism):
    """
    Mem0 memory mechanism: structured memory system based on Mem0 Platform.

    Reference Mem0 documentation:
    - Platform: https://docs.mem0.ai/platform/quickstart
    - Add Memory: https://docs.mem0.ai/core-concepts/memory-operations/add
    - Search Memory: https://docs.mem0.ai/core-concepts/memory-operations/search

    Features:
    - Automatic structured memory extraction (infer=True) or raw message storage (infer=False)
    - Automatic conflict resolution and deduplication (when infer=True)
    - Semantic retrieval + filtering + reranking
    - Uses the Mem0 Platform hosted API
    """

    def __init__(self, config: Mem0Config) -> None:
        self.config = config
        self._client: Any = None
        # Extract template title from prompt_template, used to identify enhanced messages
        # e.g.: "Based on your previous interactions, here are relevant memories:\n{memories}" -> "Based on your previous interactions, here are relevant memories:"
        self.template_title = self.config.prompt_template.split('{memories}')[0].strip()
        self._init_client()

    def _init_client(self) -> None:
        """Initialize the Mem0 Platform client."""
        if not HAS_MEM0:
            raise ImportError(
                "Mem0 Platform client not available. "
                "Please install: pip install mem0"
            )
        if not self.config.api_key:
            raise ValueError("Mem0 Platform requires api_key in config")
        self._client = MemoryClient(api_key=self.config.api_key)
        print(f"[Mem0Memory] Initialized Platform client with user_id={self.config.user_id}")

    def _extract_query(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract the first user message from messages as the retrieval query.
        Must strip injected memory content and return only the original question.
        """
        template_titles = [self.template_title]
        return extract_original_question(messages, where=self.config.where, template_titles=template_titles)

    def _format_memories(self, memories: Any) -> str:
        """Format retrieved memories as text."""
        if not memories:
            return ""

        # mem0 API may return {"results": [...]} dict or a plain list
        if isinstance(memories, dict):
            # Dict returned: try to extract results key
            if "results" in memories:
                memories = memories["results"]
            else:
                # No results key: wrap dict in a list
                memories = [memories]
        elif isinstance(memories, str):
            # String returned: return as-is
            return memories
        elif not isinstance(memories, (list, tuple)):
            # Other types: wrap in a list
            memories = [memories]

        formatted = []
        for mem in memories:
            # Ensure mem is a dict
            if isinstance(mem, dict):
                # mem0 memory format: {"memory": "...", "metadata": {...}, ...}
                memory_text = mem.get("memory", "") or mem.get("content", "")
                if memory_text:
                    formatted.append(f"- {memory_text}")
            elif isinstance(mem, str):
                # If mem is a string, use it directly
                formatted.append(f"- {mem}")

        return "\n".join(formatted)

    def _inject_memories(
        self,
        messages: List[Dict[str, Any]],
        memory_text: str
    ) -> List[Dict[str, Any]]:
        """Inject memory into messages, appended to the first user message."""
        if not memory_text:
            return list(messages) if messages is not None else []

        # Format memory content
        memory_content = self.config.prompt_template.format(memories=memory_text)

        # Inject memory using the shared utility
        return enhance_messages_with_memory(messages, memory_content, where=self.config.where)

    def use_memory(
        self,
        task: str,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on the current task and original messages, then inject them.
        """
        enhanced = list(messages) if messages is not None else []

        # Extract query
        query = self._extract_query(messages)
        if not query:
            return enhanced

        try:
            # Call mem0.search() to retrieve memories (using user-defined user_id)
            # mem0 API requires filters to include user_id; empty dict is not allowed
            # Reference docs: https://docs.mem0.ai/platform/quickstart
            search_kwargs = {
                "query": query,
                "user_id": self.config.user_id,
                "top_k": self.config.top_k,
                "filters": {"user_id": self.config.user_id},  # filters must include user_id
            }
            if self.config.threshold is not None:
                search_kwargs["threshold"] = self.config.threshold
            if self.config.rerank:
                search_kwargs["rerank"] = True

            memories = self._client.search(**search_kwargs)

            # Format memory text
            memory_text = self._format_memories(memories)

            # Inject into messages
            return self._inject_memories(enhanced, memory_text)

        except Exception as e:
            print(f"[Mem0Memory] Search failed: {e}, returning original messages")
            return enhanced

    def update_memory(
        self,
        task: str,
        history: List[Dict[str, Any]],
        result: Dict[str, Any]
    ) -> None:
        """
        Called after a single sample finishes. Writes the new trajectory/result to Mem0.
        Retries on failure according to config until success or max retries reached.
        """
        finish = result.get("finish", False)
        status = result.get("status", "")
        reward = result.get("reward", 0)
        # success_only only checks task completion (finish or status), not reward
        is_success = finish or status == "completed"

        # Filter: if success_only=True, only store successfully completed samples (regardless of reward)
        if self.config.success_only and not is_success:
            print(f"[Mem0] Skipping memory storage: success_only=True but sample not completed (finish={finish}, status={status})")
            return

        # Filter: if reward_bigger_than_zero=True, only store samples with reward > 0
        if self.config.reward_bigger_than_zero:
            if reward <= 0:
                print(f"[Mem0] Skipping memory storage: reward_bigger_than_zero=True but reward={reward}")
                return

        metadata = {
            "task": task,
            "success": is_success,  # uses computed is_success (finish or status=="completed")
        }

        # Convert history to a serializable format (filter RewardHistoryItem, convert Pydantic models)
        serialized_history = _serialize_history(history, self.template_title, self.config.where)

        # Filter and normalise messages: Mem0 API requires each message to have role and non-empty content
        # Ensure message format strictly matches Mem0 requirements (only role and content fields)
        # Mem0 API only accepts role "user" or "assistant"; "system"/"tool" etc. are not accepted
        # Conversion rules:
        # - system and tool → user
        # - assistant (including those with tool_calls) → assistant
        # - tool_calls are converted to text and merged into content
        filtered_messages = []
        for msg in serialized_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")  # check for tool_calls field

            # Ensure role exists
            if not role or not isinstance(role, str):
                continue

            # For messages with tool_calls, convert them to text and merge into content
            # Mem0 API does not support the tool_calls field; convert to text
            if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                # Convert tool_calls to text descriptions
                tool_calls_text = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func_name = tc.get("function", {}).get("name", "") if isinstance(tc.get("function"), dict) else ""
                        func_args = tc.get("function", {}).get("arguments", "") if isinstance(tc.get("function"), dict) else ""
                        if func_name:
                            tool_calls_text.append(f"Tool call: {func_name}({func_args})")
                if tool_calls_text:
                    tool_calls_str = "\n".join(tool_calls_text)
                    # Append tool_calls info to content
                    if content and str(content).strip():
                        content = f"{content}\n{tool_calls_str}"
                    else:
                        content = tool_calls_str

            # Ensure content is present and non-empty (tool_calls may have been converted to content)
            if not content or not isinstance(content, str) or not str(content).strip():
                continue

            # Mem0 API only accepts "user" and "assistant" roles
            # Conversion: system and tool → user, assistant → assistant
            role_lower = str(role).strip().lower()
            if role_lower == "assistant":
                # Keep assistant as assistant (tool_calls have already been merged into content)
                role_lower = "assistant"
            elif role_lower in ("system", "tool"):
                # Convert system and tool to user
                if role_lower == "system":
                    print(f"[Mem0Memory] Converting system message to user role")
                else:
                    print(f"[Mem0Memory] Converting tool message to user role")
                role_lower = "user"
            elif role_lower == "user":
                # Keep user as user
                role_lower = "user"
            else:
                # Unknown role: log a warning and convert to "user"
                print(f"[Mem0Memory] Unknown role '{role}', converting to 'user'")
                role_lower = "user"

            # Normalise message format: keep only role and content (remove extra fields like tool_calls, function_call, etc.)
            # Per Mem0 docs, messages should be [{"role": "user", "content": "..."}, ...]
            filtered_messages.append({
                "role": role_lower,  # use the converted role
                "content": str(content).strip()
            })

        if not filtered_messages:
            print(f"[Mem0] Skipping memory storage: No valid messages in history after filtering for task={task}, user_id={self.config.user_id}")
            return

        # Add detailed debug logging
        print(
            f"[Mem0Memory] Attempting to add memory: task={task}, user_id={self.config.user_id}, "
            f"is_success={is_success}, reward={reward}, history_length={len(history)}, "
            f"serialized_length={len(serialized_history)}, filtered_length={len(filtered_messages)}"
        )
        print(
            f"[Mem0Memory] First 3 filtered messages: {filtered_messages[:3] if len(filtered_messages) >= 3 else filtered_messages}"
        )
        print(f"[Mem0Memory] Metadata: {metadata}, infer: {self.config.infer}")

        # Retry logic: retry until success (if max_retries=-1) or until max retries is reached
        import time
        retry_count = 0
        current_delay = self.config.retry_delay

        while True:
            try:
                # Call mem0.add() to store memory (using user-defined user_id)
                # Per Mem0 docs (https://docs.mem0.ai/core-concepts/memory-operations/add):
                # - messages: required, format [{"role": "user", "content": "..."}, ...]
                # - user_id: required
                # - metadata: optional, used for filtering and retrieval
                # - infer: optional, controls whether structured memory is extracted (default True)

                # Log the actual data being sent (for debugging)
                print(
                    f"[Mem0Memory] Sending to Mem0 API: "
                    f"messages_count={len(filtered_messages)}, user_id={self.config.user_id}, "
                    f"metadata={metadata}, infer={self.config.infer}"
                )

                add_result = self._client.add(
                    messages=filtered_messages,
                    user_id=self.config.user_id,
                    metadata=metadata,
                    infer=self.config.infer,
                )


                # Check return value to confirm success
                if add_result and "results" in add_result:
                    num_memories = len(add_result["results"])
                    if num_memories > 0:
                        # Success: results returned
                        print(
                            f"[Mem0Memory] Successfully added {num_memories} memory(ies) "
                            f"for task={task}, user_id={self.config.user_id}"
                        )
                        # Wait as configured to avoid request bursts
                        if self.config.wait_time > 0:
                            print(f"[Mem0] Waiting {self.config.wait_time}s after successful add (task={task}, user_id={self.config.user_id})")
                            print(
                                f"[Mem0Memory] Waiting {self.config.wait_time}s after successful add "
                                f"(task={task}, user_id={self.config.user_id})"
                            )
                            time.sleep(self.config.wait_time)
                            print(f"[Mem0] Wait completed, continuing...")
                        return
                    else:
                        # Unexpected return: results is empty
                        print(
                            f"[Mem0Memory] Add returned empty results for task={task}, "
                            f"user_id={self.config.user_id}, result={add_result}"
                        )
                        # Continue retrying
                else:
                    # Unexpected return: no results field
                    print(
                        f"[Mem0Memory] Add returned unexpected result for task={task}, "
                        f"user_id={self.config.user_id}, result={add_result}"
                    )
                    # Continue retrying

            except Exception as e:
                # Check for non-recoverable errors (should not retry)
                error_type = type(e).__name__
                error_str = str(e)

                # Identify network/connection errors (retryable)
                is_network_error = (
                    "disconnected" in error_str.lower() or
                    "connection" in error_str.lower() or
                    "timeout" in error_str.lower() or
                    "network" in error_str.lower() or
                    "Server disconnected" in error_str or
                    "ConnectionError" in error_type or
                    "TimeoutError" in error_type or
                    "RequestException" in error_type
                )

                # For 400 errors, log detailed request info for debugging
                if "400" in error_str or "Validation" in error_type:
                    print(
                        f"[Mem0Memory] Validation error (400) for task={task}, user_id={self.config.user_id}: {e}"
                    )
                    print(
                        f"[Mem0Memory] Request details: "
                        f"messages_count={len(filtered_messages)}, "
                        f"first_message_role={filtered_messages[0].get('role') if filtered_messages else 'N/A'}, "
                        f"first_message_content_preview={filtered_messages[0].get('content', '')[:100] if filtered_messages else 'N/A'}, "
                        f"metadata={metadata}, infer={self.config.infer}"
                    )

                # Non-recoverable errors: authentication errors, validation errors, etc.
                if "Authentication" in error_type or "Validation" in error_type or "401" in error_str or "400" in error_str:
                    print(
                        f"[Mem0Memory] Non-retryable error for task={task}, user_id={self.config.user_id}: {e}"
                    )
                    raise  # Raise immediately without retrying

                # Recoverable errors: network errors, rate limit errors, etc.
                if is_network_error:
                    print(
                        f"[Mem0Memory] Network error detected (attempt {retry_count + 1}) "
                        f"for task={task}, user_id={self.config.user_id}: {error_type}: {error_str}"
                    )
                else:
                    print(
                        f"[Mem0Memory] Add memory failed (attempt {retry_count + 1}) "
                        f"for task={task}, user_id={self.config.user_id}: {error_type}: {error_str}"
                    )

                # Check whether to continue retrying
                if self.config.max_retries == 0:
                    # No retry
                    print(
                        f"[Mem0Memory] Add memory failed and retry is disabled "
                        f"for task={task}, user_id={self.config.user_id}"
                    )
                    return

                # Increment retry count regardless of max_retries setting
                retry_count += 1

                if self.config.max_retries > 0:
                    # Max retries limit is set
                    if retry_count >= self.config.max_retries:
                        print(
                            f"[Mem0Memory] Add memory failed after {retry_count} retries "
                            f"for task={task}, user_id={self.config.user_id}"
                        )
                        return

                # Wait then retry (exponential backoff)
                # For max_retries=-1 (unlimited), keeps retrying until success
                print(
                    f"[Mem0Memory] Retrying add memory in {current_delay:.2f}s "
                    f"(attempt {retry_count + 1}) for task={task}, user_id={self.config.user_id}"
                )
                time.sleep(current_delay)
                current_delay *= self.config.retry_backoff  # Exponential backoff


def load_mem0_from_yaml(config_path: str) -> Mem0Memory:
    """
    Load config from memory/mem0/mem0.yaml and construct a Mem0Memory instance.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw = cfg.get("mem0", {}) or {}

    api_key = str(raw.get("api_key", ""))
    user_id = str(raw.get("user_id", "default"))
    infer = bool(raw.get("infer", True))
    top_k = int(raw.get("top_k", 4))
    threshold = raw.get("threshold")
    if threshold is not None:
        threshold = float(threshold)
    rerank = bool(raw.get("rerank", True))
    success_only = bool(raw.get("success_only", True))
    prompt_template = raw.get(
        "prompt_template",
        "Based on your previous interactions, here are relevant memories:\n{memories}"
    )
    where = raw.get("where", "tail")
    # Retry configuration
    max_retries = int(raw.get("max_retries", -1))  # -1: unlimited retries (keep retrying until success)
    retry_delay = float(raw.get("retry_delay", 1.0))  # Retry delay (seconds)
    retry_backoff = float(raw.get("retry_backoff", 2.0))  # Exponential backoff multiplier
    # reward_bigger_than_zero config
    reward_bigger_than_zero = bool(raw.get("reward_bigger_than_zero", False))
    # Wait configuration
    wait_time = float(raw.get("wait_time", 0.0))  # Time to wait after each successful memory add (seconds)

    config = Mem0Config(
        api_key=api_key,
        user_id=user_id,
        infer=infer,
        top_k=top_k,
        threshold=threshold,
        rerank=rerank,
        success_only=success_only,
        reward_bigger_than_zero=reward_bigger_than_zero,
        prompt_template=prompt_template,
        where=where,
        max_retries=max_retries,
        retry_delay=retry_delay,
        retry_backoff=retry_backoff,
        wait_time=wait_time,
    )

    return Mem0Memory(config)

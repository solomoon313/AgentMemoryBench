from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import ReadTimeout, Timeout

import yaml

from ..base import MemoryMechanism, parse_llm_json_response
from ..streamICL.streamICL import RAG
from src.utils.message_schema import (
    extract_message_info,
    enhance_messages_with_memory,
    extract_original_question,
)


@dataclass
class AWMProConfig:
    # Model configuration (single unified model)
    model_name: str

    # Prompt configuration
    workflow_induction_prompt: str
    workflow_management_prompt: str

    # Workflow RAG configuration (required fields)
    workflow_rag_embedding_model: str
    workflow_rag_top_k: int
    workflow_rag_order: str
    workflow_rag_seed: int
    workflow_rag_prompt_template: str
    workflow_rag_where: str  # "tail": inject memory after the user question | "front": inject before
    workflow_rag_success_only: bool
    workflow_rag_reward_bigger_than_zero: bool

    # Workflow management configuration (required fields)
    workflow_management_similarity_top_k: int  # number of similar existing workflows to retrieve per new workflow

    # Workflow storage path (required field)
    workflow_storage_path: Path

    # Optional fields (fields with defaults must come last)
    workflow_induction_max_retries: int = 5  # max retries for workflow induction, default 5
    workflow_management_max_retries: int = 5  # max retries for workflow management, default 5


class AWMPro(MemoryMechanism):
    """
    Agent Workflow Memory Pro (AWMPro):
    - Focused on workflow extraction and management
    - Extracts reusable workflows from task trajectories
    - Uses RAG for workflow retrieval and management
    """

    def __init__(self, config: AWMProConfig) -> None:
        self.config = config
        self._workflow_storage_path = self.config.workflow_storage_path
        self._workflow_storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract template title from workflow_rag_prompt_template (used to identify memory-augmented messages)
        # e.g. "Here are some useful workflows:\n{workflows}" -> "Here are some useful workflows:"
        self.template_title = self.config.workflow_rag_prompt_template.split('{workflows}')[0].strip()
        self.where = self.config.workflow_rag_where

        # Initialize workflow RAG
        self._workflow_rag: Optional[RAG] = None
        try:
            self._workflow_rag = RAG(
                embedding_model=self.config.workflow_rag_embedding_model,
                top_k=self.config.workflow_rag_top_k,
                order=self.config.workflow_rag_order,
                seed=self.config.workflow_rag_seed,
            )
        except ImportError as e:
            print(f"[AWMPro] Failed to init workflow RAG: {e}. Vector retrieval disabled.")
            self._workflow_rag = None

        # Cache LLM config to avoid re-reading from disk on every call
        self._agent_cfg: Optional[Dict[str, Any]] = self._load_agent_config(self.config.model_name)
        try:
            root_dir = Path(__file__).resolve().parents[2]
            _agent_cfg_path = root_dir / "configs" / "llmapi" / "agent.yaml"
            with _agent_cfg_path.open("r", encoding="utf-8") as _f:
                self._available_models: List[str] = list((yaml.safe_load(_f) or {}).keys())
        except Exception:
            self._available_models = []

    def _load_agent_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load LLM HTTP config:
        - Assembles url / headers / body for the given model from configs/llmapi/api.yaml + agent.yaml.
        """
        model_name = (model_name or "").strip()  # strip removes leading/trailing whitespace and newlines
        if not model_name:
            return None

        # Config directory relative to project root
        root_dir = Path(__file__).resolve().parents[2]
        llmapi_dir = root_dir / "configs" / "llmapi"
        agent_cfg_path = llmapi_dir / "function_agent.yaml"
        api_cfg_path = llmapi_dir / "function_api.yaml"

        if not agent_cfg_path.exists() or not api_cfg_path.exists():
            print(
                f"[AWMPro] LLM config files not found: {agent_cfg_path}, {api_cfg_path}"
            )
            return None

        try:
            with agent_cfg_path.open("r", encoding="utf-8") as f:
                agents_cfg = yaml.safe_load(f) or {}
            if model_name not in agents_cfg:
                print(
                    f"[AWMPro] Model '{model_name}' not found in {agent_cfg_path}"
                )
                return None
            agent_cfg = agents_cfg[model_name] or {}

            with api_cfg_path.open("r", encoding="utf-8") as f:
                api_cfg = yaml.safe_load(f) or {}

            base_params = api_cfg.get("parameters", {}) or {}
            agent_params = agent_cfg.get("parameters", {}) or {}

            body = dict(base_params.get("body", {}) or {})
            body.update(agent_params.get("body", {}) or {})  # agent_params body overrides base_params body

            url = base_params.get("url") or api_cfg.get("parameters", {}).get("url")
            if not url:
                print("[AWMPro] URL not found in api.yaml / agent.yaml")
                return None

            headers = dict(base_params.get("headers", {}) or {})
            headers.update(agent_params.get("headers", {}) or {})  # agent_params headers override base_params headers

            return {"url": url, "headers": headers, "body": body}
        except Exception as e:
            print(f"[AWMPro] failed to load agent config: {e}")
            return None

    def _call_llm(self, model_name: str, messages: List[Dict[str, Any]], max_retries: int = 3, purpose: str = "LLM call") -> Optional[str]:
        """Call LLM API; supports infinite retry (max_retries=-1)"""
        print(f"[AWMPro] Calling {purpose} with model={model_name}, messages_count={len(messages)}")
        cfg = self._agent_cfg
        if not cfg:
            print(f"[AWMPro] ERROR: Failed to load agent config for model={model_name}")
            return None

        url = cfg["url"]
        headers = cfg["headers"]
        base_body = cfg["body"]

        # Log request info (excluding full messages to keep logs concise)
        print(f"[AWMPro] {purpose} request: url={url}, model={model_name}, body_keys={list(base_body.keys())}")

        body: Dict[str, Any] = {**(base_body or {}), "messages": messages}

        # Retry logic: 429/500/timeout/network errors retry with linear backoff (max 60s);
        # 400 and other non-retryable errors return immediately
        attempt = 0
        infinite_retry = (max_retries == -1)

        while infinite_retry or attempt < max_retries:
            try:
                # Per-request timeout of 250s to avoid blocking on a single sample
                resp = requests.post(url, headers=headers, json=body, timeout=250)

                # 400 Bad Request is usually a malformed request; do not retry
                # But if it's a token limit error, handle it specially
                if resp.status_code == 400:
                    try:
                        error_detail = resp.json()
                        error_message = str(error_detail.get("message", "")) if isinstance(error_detail, dict) else str(error_detail)
                        # Check if this is a token limit error
                        if "max_total_tokens" in error_message or "max_seq_len" in error_message or "exceeds" in error_message.lower():
                            print(f"[AWMPro] ERROR: LLM 400 Bad Request - Token limit exceeded (model={model_name}, purpose={purpose})")
                            print(f"[AWMPro] Error detail: {error_detail}")
                            # Return a sentinel string so the caller knows it was a token limit error
                            return "__TOKEN_LIMIT_EXCEEDED__"
                    except:
                        pass
                    # Other 400 errors
                    try:
                        error_detail = resp.json()
                    except:
                        error_detail = resp.text[:500]
                    print(f"[AWMPro] ERROR: LLM 400 Bad Request (model={model_name}, purpose={purpose})")
                    print(f"[AWMPro] Error detail: {error_detail}")
                    print(f"[AWMPro] Request body model field: {body.get('model', 'NOT SET')}")
                    if self._available_models:
                        print(f"[AWMPro] Available models in agent.yaml: {self._available_models}")
                    return None

                # 429 Too Many Requests / 500: retry with linear backoff (max 60s)
                # Linear backoff: 5*(attempt+1) seconds (5, 10, 15, ..., 60, 60, ...), capped at 60s
                if resp.status_code in (429, 500):
                    wait_sec = min(5 * (attempt + 1), 60)
                    retry_info = "infinite retries" if infinite_retry else f"{attempt + 1}/{max_retries}"
                    print(
                        f"[AWMPro] LLM HTTP {resp.status_code} (attempt {retry_info}), "
                        f"retrying after {wait_sec}s (linear backoff, max 60s)..."
                    )
                    time.sleep(wait_sec)
                    attempt += 1
                    continue

                # Other HTTP errors (e.g. 401, 403): raise immediately
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    print(f"[AWMPro] WARNING: {purpose} returned no choices in response")
                    return None
                message = choices[0].get("message") or {}
                content = (message.get("content") or "").strip()
                if content:
                    print(f"[AWMPro] {purpose} succeeded, response_length={len(content)}")
                else:
                    print(f"[AWMPro] WARNING: {purpose} returned empty content")
                return content or None

            except (ReadTimeout, Timeout) as e:
                # Timeout: retry with linear backoff (max 60s)
                # Linear backoff: 5*(attempt+1) seconds (5, 10, 15, ..., 60, 60, ...), capped at 60s
                wait_sec = min(5 * (attempt + 1), 60)
                retry_info = "infinite retries" if infinite_retry else f"{attempt + 1}/{max_retries}"
                print(
                    f"[AWMPro] LLM timeout (attempt {retry_info}), "
                    f"retrying after {wait_sec}s (linear backoff, max 60s)..."
                )
                time.sleep(wait_sec)
                attempt += 1
                continue

            except requests.exceptions.RequestException as e:
                # Other network errors (e.g. connection error): retry with linear backoff (max 60s)
                # Linear backoff: 5*(attempt+1) seconds (5, 10, 15, ..., 60, 60, ...), capped at 60s
                wait_sec = min(5 * (attempt + 1), 60)
                retry_info = "infinite retries" if infinite_retry else f"{attempt + 1}/{max_retries}"
                print(
                    f"[AWMPro] LLM network error (attempt {retry_info}): {str(e)}, "
                    f"retrying after {wait_sec}s (linear backoff, max 60s)..."
                )
                time.sleep(wait_sec)
                attempt += 1
                continue

            except Exception as e:
                # Unexpected error, do not retry
                print(f"[AWMPro] LLM fatal error: {e}")
                return None

        # All retries exhausted (only reached in non-infinite-retry mode)
        print(f"[AWMPro] ERROR: {purpose} failed after {max_retries} attempts")
        return None

    def _parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse JSON using the shared fault-tolerant parser (3-layer fallback).

        Args:
            response: raw text returned by the LLM

        Returns:
            Parsed JSON dict, or None on failure

        Fault-tolerance layers:
            1. Output cleaning: strip markdown, comments, etc.
            2. Smart repair: json_repair rescues ~90% of malformed responses
            3. Schema validation: skipped (AWMPro does not need strict schema)
        """
        result = parse_llm_json_response(
            response_text=response,
            schema=None,  # AWMPro does not need strict schema validation
            logger_prefix="AWMPro"
        )

        if result is None:
            print(f"[AWMPro] JSON parsing failed after all attempts")
            return None

        return result

    def _call_workflow_induction(self, trajectory_text: str) -> Optional[str]:
        """Call the Workflow Induction Model to extract workflows (limited retries)"""
        print(f"[AWMPro] Calling workflow induction, trajectory_text_length={len(trajectory_text)}")
        prompt = self.config.workflow_induction_prompt

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    "Here is one completed trajectory. Please extract concise reusable workflow(s) from it.\n\n"
                    f"{trajectory_text}"
                ),
            },
        ]

        attempt = 0
        max_retries = self.config.workflow_induction_max_retries
        while attempt < max_retries:
            attempt += 1
            print(f"[AWMPro] Workflow induction attempt {attempt}/{max_retries}")

            response = self._call_llm(self.config.model_name, messages, max_retries=-1, purpose="workflow induction")
            if response and len(response.strip()) > 0:
                # Check that response contains at least one workflow section (## workflow_name)
                if "## " in response:
                    print(f"[AWMPro] Workflow induction succeeded (attempt {attempt})")
                    return response
                else:
                    print(f"[AWMPro] WARNING: Workflow induction response doesn't contain workflow format (attempt {attempt})")
                    if attempt < max_retries:
                        wait_sec = min(5 * attempt, 60)
                        print(f"[AWMPro] Retrying workflow induction after {wait_sec}s...")
                        time.sleep(wait_sec)
                    continue
            else:
                if attempt < max_retries:
                    wait_sec = min(5 * attempt, 60)
                    print(f"[AWMPro] Workflow induction returned empty response, retrying after {wait_sec}s...")
                    time.sleep(wait_sec)
                continue

        print(f"[AWMPro] ERROR: Workflow induction failed after {max_retries} attempts")
        return None

    def _parse_new_workflows(self, new_workflows_text: str) -> List[str]:
        """Parse individual workflows from the induction output text"""
        workflows = []
        # Split on ## workflow_name markers
        parts = new_workflows_text.split("## ")
        for part in parts:
            part = part.strip()
            if part:
                workflows.append(part)
        return workflows

    def _find_similar_workflows(
        self, new_workflow_text: str, existing_workflows: List[Dict[str, Any]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Find existing workflows similar to a new workflow using vector search (following mem0's approach).

        Following mem0's add method:
        1. For each new workflow, use vector search to find top-k similar existing workflows
        2. Pass only similar workflows to the LLM instead of the full list
        """
        if not self._workflow_rag or not existing_workflows:
            return []

        try:
            # Retrieve similar workflows via RAG (returns workflow texts)
            retrieved_texts = self._workflow_rag.retrieve(query=new_workflow_text, top_k=top_k)
            if not retrieved_texts:
                return []

            # Map retrieved texts back to workflow objects
            # RAG stores key=value=workflow text, so retrieved text should match directly
            similar_workflows = []

            # Build text→workflow mapping for fast lookup
            text_to_workflow = {}
            for wf in existing_workflows:
                wf_text = wf.get("text", "").strip()
                if wf_text:
                    # Normalize whitespace before using as key
                    normalized_text = " ".join(wf_text.split())
                    text_to_workflow[normalized_text] = wf

            # Match retrieved texts against existing workflows
            for retrieved_text in retrieved_texts:
                retrieved_text = retrieved_text.strip()
                if not retrieved_text:
                    continue

                # Try exact match first
                normalized_retrieved = " ".join(retrieved_text.split())
                if normalized_retrieved in text_to_workflow:
                    similar_workflows.append(text_to_workflow[normalized_retrieved])
                    continue

                # Exact match failed; fall back to fuzzy matching
                best_match = None
                best_similarity = 0.0

                for wf_text_normalized, wf in text_to_workflow.items():
                    # Compute simple token-based text similarity
                    retrieved_words = set(normalized_retrieved.lower().split())
                    wf_words = set(wf_text_normalized.lower().split())

                    if not retrieved_words or not wf_words:
                        continue

                    # Jaccard similarity
                    intersection = len(retrieved_words & wf_words)
                    union = len(retrieved_words | wf_words)
                    similarity = intersection / union if union > 0 else 0.0

                    # Also check substring containment (looser match)
                    if normalized_retrieved in wf_text_normalized or wf_text_normalized in normalized_retrieved:
                        similarity = max(similarity, 0.5)  # treat containment as at least 50% similarity

                    if similarity > best_similarity and similarity > 0.3:  # 30% similarity threshold
                        best_similarity = similarity
                        best_match = wf

                if best_match and best_match not in similar_workflows:
                    similar_workflows.append(best_match)

            # Deduplicate while preserving order
            seen_ids = set()
            unique_workflows = []
            for wf in similar_workflows:
                wf_id = wf.get("id", "")
                if wf_id and wf_id not in seen_ids:
                    seen_ids.add(wf_id)
                    unique_workflows.append(wf)

            return unique_workflows[:top_k]  # cap at top_k
        except Exception as e:
            print(f"[AWMPro] Failed to find similar workflows using RAG: {e}")
            import traceback
            print(traceback.format_exc())
            return []


    def _call_workflow_management(
        self, existing_workflows: List[Dict[str, Any]], new_workflows: str
    ) -> Optional[Dict[str, Any]]:
        """Call the Workflow Management Model to ADD/UPDATE/DELETE/NONE workflows.

        Following mem0's approach:
        1. Parse new workflows from the induction output
        2. For each new workflow, find similar existing ones via vector search
        3. Pass only the similar subset to the LLM, not the full list
        4. LLM decides the operation type (ADD/UPDATE/DELETE/NONE)
        """
        # 1. Parse new workflows
        new_workflow_list = self._parse_new_workflows(new_workflows)
        if not new_workflow_list:
            print("[AWMPro] No new workflows parsed from induction result")
            return None

        print(f"[AWMPro] Parsed {len(new_workflow_list)} new workflow(s) from induction result")

        # 2. For each new workflow, find similar existing ones via vector search
        all_similar_workflows = {}  # {new_workflow_text: [similar_existing_workflows]}
        similarity_top_k = self.config.workflow_management_similarity_top_k
        for new_wf_text in new_workflow_list:
            similar_wfs = self._find_similar_workflows(new_wf_text, existing_workflows, top_k=similarity_top_k)
            all_similar_workflows[new_wf_text] = similar_wfs
            print(f"[AWMPro] Found {len(similar_wfs)} similar workflow(s) for new workflow (top_k={similarity_top_k}, first 100 chars: {new_wf_text[:100]}...)")

        # 3. Merge all similar workflows (deduplicated)
        all_similar_ids = set()
        similar_workflows_list = []
        for similar_wfs in all_similar_workflows.values():
            for wf in similar_wfs:
                wf_id = wf.get("id", "")
                if wf_id not in all_similar_ids:
                    all_similar_ids.add(wf_id)
                    similar_workflows_list.append(wf)

        # 4. Format similar workflows for the LLM (only similar subset, not full list)
        existing_text = ""
        if similar_workflows_list:
            workflow_list = []
            for wf in similar_workflows_list:
                wf_id = wf.get("id", "")
                wf_text = wf.get("text", "")
                workflow_list.append(f"ID: {wf_id}\n{wf_text}")
            existing_text = "\n\n".join(workflow_list)
            print(f"[AWMPro] Using {len(similar_workflows_list)} similar workflow(s) for LLM comparison (out of {len(existing_workflows)} total)")
        else:
            print(f"[AWMPro] No similar workflows found, all new workflows will be ADDed")
            existing_text = "Current workflow memory is empty."

        # Build prompt
        prompt = self.config.workflow_management_prompt.format(
            existing_workflows=existing_text,
            new_workflows=new_workflows
        )

        messages = [
            {"role": "user", "content": prompt}
        ]

        attempt = 0
        max_similar_to_keep = len(similar_workflows_list)  # track how many are passed to LLM, for truncation
        max_retries = self.config.workflow_management_max_retries  # max retries from config
        while attempt < max_retries:  # bounded retry loop
            attempt += 1
            print(f"[AWMPro] Workflow management attempt {attempt}/{max_retries}")

            response = self._call_llm(self.config.model_name, messages, max_retries=-1, purpose="workflow management")

            # Check for token limit error
            if response == "__TOKEN_LIMIT_EXCEEDED__":
                print(f"[AWMPro] Token limit exceeded, trying to truncate input...")
                if max_similar_to_keep > 5:
                    max_similar_to_keep = max(5, max_similar_to_keep // 2)
                    print(f"[AWMPro] Truncating similar workflows to {max_similar_to_keep}")
                    truncated_similar = similar_workflows_list[:max_similar_to_keep]
                    if truncated_similar:
                        workflow_list = []
                        for wf in truncated_similar:
                            wf_id = wf.get("id", "")
                            wf_text = wf.get("text", "")
                            workflow_list.append(f"ID: {wf_id}\n{wf_text}")
                        existing_text = "\n\n".join(workflow_list)
                    else:
                        existing_text = "Current workflow memory is empty."
                    # Rebuild prompt with truncated input
                    prompt = self.config.workflow_management_prompt.format(
                        existing_workflows=existing_text,
                        new_workflows=new_workflows
                    )
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    wait_sec = min(5 * attempt, 60)
                    print(f"[AWMPro] Retrying workflow management with truncated input after {wait_sec}s...")
                    time.sleep(wait_sec)
                    continue
                else:
                    # Already at minimum size but still over limit; skip this update
                    print(f"[AWMPro] WARNING: Token limit exceeded even with minimal workflows, skipping workflow update")
                    return None

            if not response:
                wait_sec = min(5 * attempt, 60)
                print(f"[AWMPro] Workflow management returned no response, retrying after {wait_sec}s...")
                time.sleep(wait_sec)
                continue

            # Parse JSON response
            result = self._parse_json_response(response)
            if result and "memory" in result:
                memory_ops = result.get("memory", [])
                print(f"[AWMPro] Workflow management succeeded (attempt {attempt})")
                print(f"[AWMPro] Workflow management operations: {len(memory_ops)} operation(s)")
                # Log each operation's details
                for idx, op in enumerate(memory_ops, 1):
                    op_id = op.get("id", "N/A")
                    op_event = op.get("event", "UNKNOWN").upper()
                    op_text = op.get("text", "")
                    op_old_memory = op.get("old_memory", "")
                    text_preview = op_text[:100] + "..." if len(op_text) > 100 else op_text
                    print(f"  [{idx}] {op_event}: id={op_id}, text_preview=\"{text_preview}\"")
                    if op_old_memory and op_event == "UPDATE":
                        old_preview = op_old_memory[:100] + "..." if len(op_old_memory) > 100 else op_old_memory
                        print(f"       old_memory_preview=\"{old_preview}\"")
                return result
            else:
                if result:
                    print(f"[AWMPro] WARNING: Workflow management response missing 'memory' key: {result}")
                else:
                    print(f"[AWMPro] WARNING: Failed to parse workflow management response (attempt {attempt})")
                    # Log full response for debugging (truncated to avoid overly long output)
                    response_preview = response[:2000] if len(response) > 2000 else response
                    print(f"[AWMPro] Workflow management raw response (first {len(response_preview)} chars): {response_preview}")
                    if len(response) > 2000:
                        print(f"[AWMPro] ... (truncated, total length: {len(response)} chars)")
                    # Try manual JSON parse for debugging
                    try:
                        test_result = json.loads(response.strip())
                        print(f"[AWMPro] DEBUG: Manual JSON parse succeeded! Result keys: {list(test_result.keys()) if isinstance(test_result, dict) else 'not a dict'}")
                    except Exception as e:
                        print(f"[AWMPro] DEBUG: Manual JSON parse also failed: {e}")
                        # Try stripping BOM or invisible characters
                        try:
                            cleaned_test = response.strip().encode('utf-8').decode('utf-8-sig')
                            test_result2 = json.loads(cleaned_test)
                            print(f"[AWMPro] DEBUG: JSON parse succeeded after BOM removal!")
                        except Exception as e2:
                            print(f"[AWMPro] DEBUG: JSON parse still failed after BOM removal: {e2}")
                wait_sec = min(5 * attempt, 60)
                print(f"[AWMPro] Retrying workflow management after {wait_sec}s...")
                time.sleep(wait_sec)
                continue

        # Max retries reached without success; skip this update
        print(f"[AWMPro] WARNING: Workflow management failed after {max_retries} attempts, skipping workflow update")
        return None

    def _load_workflows(self) -> List[Dict[str, Any]]:
        """Load persisted workflows from disk"""
        if not self._workflow_storage_path.exists():
            return []
        try:
            with open(self._workflow_storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return []

    def _save_workflows(self, workflows: List[Dict[str, Any]]) -> None:
        """Save workflows to disk"""
        with open(self._workflow_storage_path, "w", encoding="utf-8") as f:
            json.dump(workflows, f, ensure_ascii=False, indent=2)

    def _extract_question_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract the first user message as the retrieval query,
        stripping any injected workflow memory to return the original question.
        """
        template_titles = [self.template_title]
        return extract_original_question(messages, where=self.where, template_titles=template_titles)

    def use_memory(
        self, task: str, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Augment messages with retrieved workflow memory:
        1. Retrieve relevant workflows from RAG
        2. Inject into the first user message
        """
        # Retrieve workflow memory via RAG
        if self._workflow_rag:
            question = self._extract_question_from_messages(messages)
            if question:
                retrieved_texts = self._workflow_rag.retrieve(
                    query=question, top_k=self._workflow_rag.top_k
                )
                if retrieved_texts:
                    print(f"[AWMPro] Retrieved {len(retrieved_texts)} workflows from RAG")

                    # Format retrieved workflows
                    formatted_workflows = "\n\n".join(retrieved_texts)
                    workflow_memory_text = self.config.workflow_rag_prompt_template.format(
                        workflows=formatted_workflows
                    )

                    # Inject memory using shared utility
                    return enhance_messages_with_memory(messages, workflow_memory_text, where=self.where)

        return list(messages) if messages is not None else []

    def _build_trajectory_text(
        self, task: str, history: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Build the trajectory text for workflow induction.
        Important: strip workflow memory injected by use_memory() to keep only the original interaction.
        """
        if not history:
            return None

        template_titles = [self.template_title]
        parts: List[str] = [f"Task: {task}"]

        for msg in history:
            role, content, msg_dict = extract_message_info(msg)
            if role is None:
                continue
            content = str(content).strip() if content else ""
            if not content:
                continue

            if role == "user":
                # If the user message contains injected memory, extract the original question
                from src.utils.message_schema import ORIGINAL_QUESTION_SEPARATOR
                has_memory = (
                    ORIGINAL_QUESTION_SEPARATOR in content or
                    any(title in content for title in template_titles)
                )

                if has_memory:
                    # Extract original question using shared utility
                    question = extract_original_question([msg], where=self.where, template_titles=template_titles)
                    if question:
                        content = question
                    else:
                        content = ""

                if content:  # only append if content remains after stripping memory
                    parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "tool":
                tool_name = "tool"
                if msg_dict:
                    tool_name = msg_dict.get("name") or msg_dict.get("tool_call_id") or "tool"
                parts.append(f"Tool[{tool_name}]: {content}")

        return "\n".join(parts)

    def update_memory(
        self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]
    ) -> None:
        """
        Update workflow memory:
        1. Check filter conditions (success_only, reward_bigger_than_zero)
        2. Run Workflow Induction to extract workflows
        3. Run Workflow Management to ADD/UPDATE/DELETE/NONE
        """
        # Check filter conditions
        status = result.get("status", "")
        finish = result.get("finish", False)
        reward = result.get("reward", 0)
        is_success = finish or (status == "completed")

        print(f"[AWMPro] update_memory called: task={task}, finish={finish}, status={status}, reward={reward}, "
              f"is_success={is_success}, success_only={self.config.workflow_rag_success_only}, "
              f"reward_bigger_than_zero={self.config.workflow_rag_reward_bigger_than_zero}")

        if self.config.workflow_rag_success_only:
            if not is_success:
                print(f"[AWMPro] Skipping workflow update: success_only=True but sample not completed "
                      f"(finish={finish}, status={status})")
                return

        if self.config.workflow_rag_reward_bigger_than_zero:
            if reward <= 0:
                print(f"[AWMPro] Skipping workflow update: reward_bigger_than_zero=True but reward={reward}")
                return

        # Proceed to update workflow memory
        print("[AWMPro] Updating workflow memory...")
        self._update_system_memory(task, history)

    def _update_system_memory(
        self, task: str, history: List[Dict[str, Any]]
    ) -> None:
        """Core workflow memory update logic"""
        # Build trajectory text
        trajectory_text = self._build_trajectory_text(task, history)
        if not trajectory_text:
            return

        # Run Workflow Induction
        new_workflows_text = self._call_workflow_induction(trajectory_text)
        if not new_workflows_text:
            print("[AWMPro] Workflow induction failed")
            return

        # Load existing workflows
        existing_workflows = self._load_workflows()

        # Run Workflow Management
        management_result = self._call_workflow_management(existing_workflows, new_workflows_text)
        if not management_result:
            print("[AWMPro] Workflow management failed")
            return

        # Apply ADD/UPDATE/DELETE/NONE operations
        memory_ops = management_result.get("memory", [])
        if not memory_ops:
            print("[AWMPro] No memory operations to apply")
            return

        print(f"[AWMPro] Applying {len(memory_ops)} workflow management operation(s)...")

        # Build id → workflow dict
        workflow_dict = {wf.get("id", ""): wf for wf in existing_workflows}

        # Find the current max numeric ID (for assigning new IDs)
        max_id = -1
        for wf_id in workflow_dict.keys():
            try:
                id_num = int(wf_id)
                max_id = max(max_id, id_num)
            except ValueError:
                pass

        # Process each operation
        rag_needs_rebuild = False
        add_count = 0
        update_count = 0
        delete_count = 0
        none_count = 0

        for op in memory_ops:
            op_id = op.get("id", "")
            op_event = op.get("event", "").upper()
            op_text = op.get("text", "")

            if op_event == "ADD":
                # ADD: assign new ID (max + 1)
                new_id = str(max_id + 1)
                max_id += 1
                workflow_dict[new_id] = {"id": new_id, "text": op_text}
                # Insert into RAG (text is both key and value)
                if self._workflow_rag:
                    self._workflow_rag.insert(key=op_text, value=op_text)
                add_count += 1
                text_preview = op_text[:80] + "..." if len(op_text) > 80 else op_text
                print(f"  [ADD] Created new workflow id={new_id}, text_preview=\"{text_preview}\"")
            elif op_event == "UPDATE":
                # UPDATE: overwrite existing workflow text
                if op_id in workflow_dict:
                    old_text = workflow_dict[op_id].get("text", "")
                    workflow_dict[op_id]["text"] = op_text
                    # RAG does not support in-place update; flag for index rebuild
                    rag_needs_rebuild = True
                    update_count += 1
                    old_preview = old_text[:80] + "..." if len(old_text) > 80 else old_text
                    new_preview = op_text[:80] + "..." if len(op_text) > 80 else op_text
                    print(f"  [UPDATE] Updated workflow id={op_id}")
                    print(f"    old: \"{old_preview}\"")
                    print(f"    new: \"{new_preview}\"")
                else:
                    print(f"  [UPDATE] WARNING: Workflow id={op_id} not found, skipping update")
            elif op_event == "DELETE":
                # DELETE: remove workflow
                if op_id in workflow_dict:
                    deleted_text = workflow_dict[op_id].get("text", "")
                    del workflow_dict[op_id]
                    # RAG does not support in-place deletion; flag for index rebuild
                    rag_needs_rebuild = True
                    delete_count += 1
                    text_preview = deleted_text[:80] + "..." if len(deleted_text) > 80 else deleted_text
                    print(f"  [DELETE] Deleted workflow id={op_id}, text_preview=\"{text_preview}\"")
                else:
                    print(f"  [DELETE] WARNING: Workflow id={op_id} not found, skipping delete")
            elif op_event == "NONE":
                # NONE: no change needed
                none_count += 1
                text_preview = op_text[:80] + "..." if len(op_text) > 80 else op_text
                print(f"  [NONE] No change for workflow id={op_id}, text_preview=\"{text_preview}\"")
            else:
                print(f"  [UNKNOWN] Unknown event type: {op_event}, id={op_id}")

        # Log operation summary
        print(f"[AWMPro] Workflow management operations summary: ADD={add_count}, UPDATE={update_count}, DELETE={delete_count}, NONE={none_count}")

        # Persist updated workflows
        updated_workflows = list(workflow_dict.values())
        self._save_workflows(updated_workflows)

        # Rebuild RAG index if any UPDATE or DELETE occurred
        if rag_needs_rebuild and self._workflow_rag:
            self._rebuild_rag_index()

    def _rebuild_rag_index(self) -> None:
        """Rebuild the RAG index from workflows.json.
        Note: creates a new RAG instance to clear the old index, then re-inserts all workflows."""
        if not self._workflow_rag:
            return

        # Clear the existing index by recreating the RAG instance
        # (RAG has no clear() method, so we must recreate it)
        try:
            self._workflow_rag = RAG(
                embedding_model=self.config.workflow_rag_embedding_model,
                top_k=self.config.workflow_rag_top_k,
                order=self.config.workflow_rag_order,
                seed=self.config.workflow_rag_seed,
            )
        except Exception as e:
            print(f"[AWMPro] Failed to rebuild RAG index: {e}")
            return

        # Reload all workflows from disk and insert into fresh index
        workflows = self._load_workflows()
        print(f"[AWMPro] Rebuilding RAG index with {len(workflows)} workflow(s) from workflows.json...")
        for wf in workflows:
            wf_text = wf.get("text", "")
            if wf_text:
                self._workflow_rag.insert(key=wf_text, value=wf_text)

        print(f"[AWMPro] RAG index rebuilt successfully with {len(workflows)} workflow(s)")


def load_awmpro_from_yaml(config_path: str) -> AWMPro:
    """Load and construct an AWMPro instance from a YAML config file"""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    raw = cfg.get("awmpro", {}) or {}

    # Model configuration
    model_name = str(raw.get("model_name", ""))
    if not model_name:
        raise ValueError("AWMPro config must specify 'model_name'")

    # Prompt configuration
    workflow_induction_prompt = raw.get("workflow_induction_prompt", "") or ""
    workflow_management_prompt = raw.get("workflow_management_prompt", "") or ""

    # Workflow RAG configuration
    workflow_rag_raw = raw.get("workflow_rag", {}) or {}
    workflow_rag_embedding_model = workflow_rag_raw.get("embedding_model", "")
    workflow_rag_top_k = int(workflow_rag_raw.get("top_k", 100))
    workflow_rag_order = str(workflow_rag_raw.get("order", "similar_at_top"))
    workflow_rag_seed = int(workflow_rag_raw.get("seed", 42))
    workflow_rag_prompt_template = raw.get("prompt_template", "") or ""
    workflow_rag_where = raw.get("where", "tail")
    workflow_rag_success_only = bool(raw.get("success_only", True))
    workflow_rag_reward_bigger_than_zero = bool(raw.get("reward_bigger_than_zero", True))

    # Workflow management configuration
    workflow_management_similarity_top_k = int(raw.get("workflow_management_similarity_top_k", 5))  # default 5, following mem0

    # Max retries configuration
    workflow_induction_max_retries = int(raw.get("workflow_induction_max_retries", 5))  # default 5
    workflow_management_max_retries = int(raw.get("workflow_management_max_retries", 5))  # default 5

    # Workflow storage path
    workflow_storage_path = Path(raw.get("workflow_storage_path", "memory/awmPro/workflows.json"))

    config = AWMProConfig(
        model_name=model_name,
        workflow_induction_prompt=workflow_induction_prompt,
        workflow_management_prompt=workflow_management_prompt,
        workflow_rag_embedding_model=workflow_rag_embedding_model,
        workflow_rag_top_k=workflow_rag_top_k,
        workflow_rag_order=workflow_rag_order,
        workflow_rag_seed=workflow_rag_seed,
        workflow_rag_prompt_template=workflow_rag_prompt_template,
        workflow_rag_where=workflow_rag_where,
        workflow_rag_success_only=workflow_rag_success_only,
        workflow_rag_reward_bigger_than_zero=workflow_rag_reward_bigger_than_zero,
        workflow_management_similarity_top_k=workflow_management_similarity_top_k,
        workflow_storage_path=workflow_storage_path,
        workflow_induction_max_retries=workflow_induction_max_retries,
        workflow_management_max_retries=workflow_management_max_retries,
    )
    return AWMPro(config)

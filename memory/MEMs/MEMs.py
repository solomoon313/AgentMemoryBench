from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

from ..base import MemoryMechanism, parse_llm_json_response

logger = logging.getLogger(__name__)


@dataclass
class MemorySourceConfig:
    """Configuration for a single memory source."""
    name: str  # Memory source name, e.g. "system_memory" or "personal_memory"
    config_path: Path  # Path to the memory source config file


@dataclass
class MEMsConfig:
    """MEMs configuration."""
    model_name: str
    memory_source_1: MemorySourceConfig
    memory_source_2: MemorySourceConfig
    retrieval_trigger_prompt: str
    update_trigger_prompt: str
    trigger_model_max_retries: int = 5
    update_success_only: bool = True
    update_reward_bigger_than_zero: bool = True


class MEMs(MemoryMechanism):
    """
    Multi-Enhanced Memory System (MEMs):
    - Supports any combination of two memory sources
    - Uses a Trigger Model to decide which memory source to use at retrieval and update time
    """

    def __init__(self, config: MEMsConfig) -> None:
        self.config = config

        # Cache LLM config to avoid re-reading from disk on every trigger model call
        self._agent_cfg: Optional[Dict[str, Any]] = self._load_agent_config(config.model_name)

        # Dynamically load two memory sources
        self.memory_1: Optional[MemoryMechanism] = self._load_memory_source(
            config.memory_source_1
        )
        self.memory_2: Optional[MemoryMechanism] = self._load_memory_source(
            config.memory_source_2
        )

        if not self.memory_1 and not self.memory_2:
            raise ValueError("[MEMs] Failed to load any memory source")

        logger.info(
            f"[MEMs] Initialized with {config.memory_source_1.name} and {config.memory_source_2.name}"
        )

    def _load_memory_source(self, source_config: MemorySourceConfig) -> Optional[MemoryMechanism]:
        """Dynamically load a memory source from its config file."""
        config_path = source_config.config_path
        if not config_path.exists():
            logger.warning(f"[MEMs] Config not found: {config_path}")
            return None

        try:
            # Read the config file
            with config_path.open("r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

            # Infer memory source type from the top-level key in the config file
            if "mem0" in yaml_data:
                from ..mem0.mem0 import Mem0Memory, Mem0Config
                mem0_cfg = yaml_data["mem0"]
                memory_config = Mem0Config(
                    api_key=mem0_cfg.get("api_key", ""),
                    user_id=mem0_cfg.get("user_id", "default_user"),
                    infer=mem0_cfg.get("infer", True),
                    top_k=mem0_cfg.get("top_k", 100),
                    threshold=mem0_cfg.get("threshold"),
                    rerank=mem0_cfg.get("rerank", True),
                    success_only=mem0_cfg.get("success_only", True),
                    reward_bigger_than_zero=mem0_cfg.get("reward_bigger_than_zero", True),
                    prompt_template=mem0_cfg.get("prompt_template", ""),
                    where=mem0_cfg.get("where", "tail"),
                    max_retries=mem0_cfg.get("max_retries", -1),
                    retry_delay=mem0_cfg.get("retry_delay", 5.0),
                    retry_backoff=mem0_cfg.get("retry_backoff", 2.0),
                    wait_time=mem0_cfg.get("wait_time", 0.0),
                )
                return Mem0Memory(memory_config)

            elif "awmpro" in yaml_data:
                from ..awmPro.awmPro import AWMPro, AWMProConfig
                awm_cfg = yaml_data["awmpro"]
                rag_cfg = awm_cfg.get("workflow_rag", {}) or {}
                memory_config = AWMProConfig(
                    model_name=awm_cfg.get("model_name", ""),
                    workflow_induction_prompt=awm_cfg.get("workflow_induction_prompt", ""),
                    workflow_management_prompt=awm_cfg.get("workflow_management_prompt", ""),
                    workflow_induction_max_retries=awm_cfg.get("workflow_induction_max_retries", 5),
                    workflow_management_max_retries=awm_cfg.get("workflow_management_max_retries", 5),
                    workflow_rag_embedding_model=rag_cfg.get("embedding_model", ""),
                    workflow_rag_top_k=rag_cfg.get("top_k", 50),
                    workflow_rag_order=rag_cfg.get("order", "similar_at_top"),
                    workflow_rag_seed=rag_cfg.get("seed", 42),
                    workflow_rag_prompt_template=awm_cfg.get("prompt_template", ""),
                    workflow_rag_where=awm_cfg.get("where", "tail"),
                    workflow_rag_success_only=awm_cfg.get("success_only", True),
                    workflow_rag_reward_bigger_than_zero=awm_cfg.get("reward_bigger_than_zero", True),
                    workflow_management_similarity_top_k=awm_cfg.get("workflow_management_similarity_top_k", 5),
                    workflow_storage_path=Path(awm_cfg.get("workflow_storage_path", "memory/awmPro/workflows.jsonl")),
                )
                return AWMPro(memory_config)

            elif "streamicl" in yaml_data:
                from ..streamICL.streamICL import StreamICLMemory, StreamICLConfig
                stream_cfg = yaml_data["streamicl"]
                memory_config = StreamICLConfig(
                    embedding_model=stream_cfg.get("embedding_model", ""),
                    top_k=stream_cfg.get("top_k", 50),
                    order=stream_cfg.get("order", "similar_at_top"),
                    seed=stream_cfg.get("seed", 42),
                    prompt_template=stream_cfg.get("prompt_template", ""),
                    where=stream_cfg.get("where", "tail"),
                    success_only=stream_cfg.get("success_only", True),
                    reward_bigger_than_zero=stream_cfg.get("reward_bigger_than_zero", True),
                    storage_path=Path(stream_cfg.get("storage_path", "memory/streamICL/trajectories.jsonl")),
                )
                return StreamICLMemory(memory_config)

            else:
                logger.warning(f"[MEMs] Unknown memory type in {config_path}")
                return None

        except Exception as e:
            logger.error(f"[MEMs] Failed to load memory source from {config_path}: {e}")
            return None

    def _load_agent_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Load LLM config from agent.yaml and api.yaml."""
        model_name = (model_name or "").strip()
        if not model_name:
            return None

        root_dir = Path(__file__).resolve().parents[2]
        llmapi_dir = root_dir / "configs" / "llmapi"
        agent_cfg_path = llmapi_dir / "agent.yaml"
        api_cfg_path = llmapi_dir / "api.yaml"

        if not agent_cfg_path.exists() or not api_cfg_path.exists():
            logger.warning(f"[MEMs] LLM config files not found")
            return None

        try:
            with agent_cfg_path.open("r", encoding="utf-8") as f:
                agents_cfg = yaml.safe_load(f) or {}
            if model_name not in agents_cfg:
                logger.warning(f"[MEMs] Model '{model_name}' not found in agent.yaml")
                return None
            agent_cfg = agents_cfg[model_name] or {}

            with api_cfg_path.open("r", encoding="utf-8") as f:
                api_cfg = yaml.safe_load(f) or {}

            base_params = api_cfg.get("parameters", {}) or {}
            agent_params = agent_cfg.get("parameters", {}) or {}

            body = dict(base_params.get("body", {}) or {})
            body.update(agent_params.get("body", {}) or {})

            url = base_params.get("url") or api_cfg.get("parameters", {}).get("url")
            if not url:
                logger.warning("[MEMs] URL not found in api.yaml")
                return None

            headers = dict(base_params.get("headers", {}) or {})
            headers.update(agent_params.get("headers", {}) or {})

            return {"url": url, "headers": headers, "body": body}
        except Exception as e:
            logger.warning(f"[MEMs] Failed to load agent config: {e}")
            return None

    def _call_llm(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 3,
        purpose: str = "LLM call",
        request_overrides: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Call the LLM API."""
        logger.info(f"[MEMs] {purpose}: messages_count={len(messages)}")

        cfg = self._agent_cfg
        if not cfg:
            logger.error(f"[MEMs] Failed to load agent config")
            return None

        url = cfg["url"]
        headers = cfg["headers"]
        base_body = cfg["body"]
        body: Dict[str, Any] = {
            **base_body,
            "messages": messages,
            **(request_overrides or {}),
        }

        attempt = 0
        while max_retries == -1 or attempt < max_retries:
            attempt += 1
            try:
                response = requests.post(
                    url, headers=headers, json=body, timeout=120
                )
                response.raise_for_status()
                data = response.json()

                content = None
                if "choices" in data and len(data["choices"]) > 0:
                    content = data["choices"][0].get("message", {}).get("content")
                elif "data" in data and "response" in data["data"]:
                    content = data["data"]["response"]

                if content:
                    return content

                logger.warning(f"[MEMs] {purpose} attempt {attempt}: No content in response")

            except Exception as e:
                logger.warning(f"[MEMs] {purpose} attempt {attempt} failed: {e}")

        logger.error(f"[MEMs] {purpose} failed after {attempt} attempts")
        return None

    def _call_trigger_model(
        self,
        prompt_template: str,
        **kwargs
    ) -> Optional[List[str]]:
        """
        Call the trigger model to decide which memory sources to use.
        Returns: ["system_memory"], ["personal_memory"], ["system_memory", "personal_memory"], or None
        """
        prompt = prompt_template.format(**kwargs)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        response = self._call_llm(
            messages=messages,
            max_retries=self.config.trigger_model_max_retries,
            purpose="Trigger Model",
            request_overrides={"response_format": {"type": "json_object"}},
        )

        if not response:
            return None

        # Parse JSON response
        parsed = parse_llm_json_response(response)
        if not parsed or "sources" not in parsed:
            logger.warning(f"[MEMs] Trigger model response invalid: {response}")
            return None

        sources = parsed["sources"]
        if not isinstance(sources, list):
            logger.warning(f"[MEMs] Trigger model sources not a list: {sources}")
            return None

        logger.info(f"[MEMs] Trigger model decision: {sources}, reasoning: {parsed.get('reasoning', 'N/A')}")
        return sources

    def use_memory(
        self,
        task: str,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Augment messages with retrieved memories:
        1. Call trigger model to decide which memory source(s) to retrieve from
        2. Retrieve memories from selected source(s), chaining the output sequentially
        3. Return the augmented messages
        """
        # Extract query for the trigger model (use first user message, fall back to task)
        query = task
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                query = msg.get("content", task)
                break

        # Call retrieval trigger model
        sources = self._call_trigger_model(
            self.config.retrieval_trigger_prompt,
            query=query
        )

        if not sources:
            logger.warning("[MEMs] Trigger model failed, using both sources as fallback")
            print(f"[MEMs] Retrieval trigger model failed — falling back to both sources: "
                  f"{self.config.memory_source_1.name}, {self.config.memory_source_2.name}")
            sources = [self.config.memory_source_1.name, self.config.memory_source_2.name]
        else:
            print(f"[MEMs] Retrieval trigger model selected sources: {sources}")

        # Retrieve from selected sources, chaining results sequentially
        enhanced_messages = list(messages)

        if self.config.memory_source_1.name in sources and self.memory_1:
            print(f"[MEMs] Retrieving from {self.config.memory_source_1.name} ...")
            enhanced_messages = self.memory_1.use_memory(task, enhanced_messages)
            print(f"[MEMs] Done retrieving from {self.config.memory_source_1.name}")

        if self.config.memory_source_2.name in sources and self.memory_2:
            print(f"[MEMs] Retrieving from {self.config.memory_source_2.name} ...")
            enhanced_messages = self.memory_2.use_memory(task, enhanced_messages)
            print(f"[MEMs] Done retrieving from {self.config.memory_source_2.name}")

        active = [s for s in [self.config.memory_source_1.name, self.config.memory_source_2.name] if s in sources]
        skipped = [s for s in [self.config.memory_source_1.name, self.config.memory_source_2.name] if s not in sources]
        print(f"[MEMs] use_memory complete — used: {active}, skipped: {skipped}")
        return enhanced_messages

    def update_memory(
        self,
        task: str,
        history: List[Dict[str, Any]],
        result: Dict[str, Any]
    ) -> None:
        """
        Update memory sources:
        1. Check update conditions (success_only, reward > 0)
        2. Call trigger model to decide which source(s) to update
        3. Update the selected source(s)
        """
        # Check update conditions
        if self.config.update_success_only:
            status = result.get("status", "")
            reward = result.get("reward", 0)
            is_success = status == "completed" or reward > 0
            if not is_success:
                logger.info("[MEMs] Skipping update: task not successful")
                print(f"[MEMs] Skipping update: update_success_only=True but task not successful "
                      f"(status={status}, reward={reward})")
                return

        if self.config.update_reward_bigger_than_zero:
            reward = result.get("reward", 0)
            if reward <= 0:
                logger.info("[MEMs] Skipping update: reward <= 0")
                print(f"[MEMs] Skipping update: reward_bigger_than_zero=True but reward={reward}")
                return

        # Serialise history for the trigger model prompt; handle Pydantic/non-serialisable objects
        def _default(obj: Any) -> Any:
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            if hasattr(obj, "dict"):
                return obj.dict()
            return str(obj)

        history_text = json.dumps(history, ensure_ascii=False, indent=2, default=_default)

        # Call update trigger model
        sources = self._call_trigger_model(
            self.config.update_trigger_prompt,
            history=history_text
        )

        if not sources:
            logger.warning("[MEMs] Trigger model failed, skipping update")
            print("[MEMs] Update trigger model failed — skipping update entirely")
            return

        print(f"[MEMs] Update trigger model selected sources: {sources}")

        # Update selected sources
        if self.config.memory_source_1.name in sources and self.memory_1:
            print(f"[MEMs] Updating {self.config.memory_source_1.name} ...")
            self.memory_1.update_memory(task, history, result)
            print(f"[MEMs] Done updating {self.config.memory_source_1.name}")

        if self.config.memory_source_2.name in sources and self.memory_2:
            print(f"[MEMs] Updating {self.config.memory_source_2.name} ...")
            self.memory_2.update_memory(task, history, result)
            print(f"[MEMs] Done updating {self.config.memory_source_2.name}")

        active = [s for s in [self.config.memory_source_1.name, self.config.memory_source_2.name] if s in sources]
        skipped = [s for s in [self.config.memory_source_1.name, self.config.memory_source_2.name] if s not in sources]
        print(f"[MEMs] update_memory complete — updated: {active}, skipped: {skipped}")

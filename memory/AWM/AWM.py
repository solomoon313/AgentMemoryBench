from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

import yaml
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from memory.base import MemoryMechanism
from memory.streamICL.streamICL import RAG
from src.runner.agent import _extract_api_key, _normalize_base_url
from src.utils.message_schema import (
    assert_memory_injection_position,
    enhance_messages_with_memory,
    extract_message_info,
    extract_original_question,
)


LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[2]
LLMAPI_DIR = ROOT_DIR / "configs" / "llmapi"


@dataclass
class AWMConfig:
    model_name: str
    instruction_prompt_path: Path
    one_shot_prompt_path: Path
    workflow_storage_path: Path
    prompt_template: str
    where: str
    success_only: bool
    reward_bigger_than_zero: bool
    workflow_rag_embedding_model: str
    workflow_rag_top_k: int
    workflow_rag_order: str
    workflow_rag_seed: int
    max_retries: int = 3


class AWM(MemoryMechanism):
    """
    A lightweight Agent Workflow Memory implementation.

    Core idea:
    - update_memory(): induce reusable tool-centric workflows from completed trajectories
    - use_memory(): retrieve relevant workflows and inject them into the current prompt
    """

    def __init__(self, config: AWMConfig) -> None:
        self.config = config
        self.where = config.where
        self.template_title = self.config.prompt_template.split("{workflows}")[0].strip()

        self.instruction_prompt = self._read_text(self.config.instruction_prompt_path)
        self.one_shot_prompt = self._read_text(self.config.one_shot_prompt_path)

        self.config.workflow_storage_path.parent.mkdir(parents=True, exist_ok=True)

        self.base_url, self.api_key, self.base_body = self._load_llm_config(self.config.model_name)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        self.rag: Optional[RAG] = None
        try:
            self.rag = RAG(
                embedding_model=self.config.workflow_rag_embedding_model,
                top_k=self.config.workflow_rag_top_k,
                order=self.config.workflow_rag_order,
                seed=self.config.workflow_rag_seed,
            )
        except Exception as exc:
            LOGGER.warning("[AWM] Failed to initialize workflow RAG: %s", exc)
            self.rag = None

        self._bootstrap_rag()

    @staticmethod
    def _read_text(path: Path) -> str:
        with path.open("r", encoding="utf-8") as f:
            return f.read().strip()

    @staticmethod
    def _resolve_path(path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path

    @staticmethod
    def _load_llm_config(model_name: str) -> Tuple[str, str, Dict[str, Any]]:
        agent_cfg_path = LLMAPI_DIR / "function_agent.yaml"
        api_cfg_path = LLMAPI_DIR / "function_api.yaml"

        with agent_cfg_path.open("r", encoding="utf-8") as f:
            agents_cfg = yaml.safe_load(f) or {}
        if model_name not in agents_cfg:
            raise ValueError(f"Model '{model_name}' not found in {agent_cfg_path}")

        with api_cfg_path.open("r", encoding="utf-8") as f:
            api_cfg = yaml.safe_load(f) or {}

        base_params = api_cfg.get("parameters", {}) or {}
        agent_params = (agents_cfg.get(model_name) or {}).get("parameters", {}) or {}

        body = dict(base_params.get("body", {}) or {})
        body.update(agent_params.get("body", {}) or {})

        url = base_params.get("url")
        if not url:
            raise ValueError("URL not found in function_api.yaml")

        headers = dict(base_params.get("headers", {}) or {})
        headers.update(agent_params.get("headers", {}) or {})

        api_key = _extract_api_key(headers)
        base_url = _normalize_base_url(url)
        return base_url, api_key, body

    def _call_llm(self, messages: List[Dict[str, Any]]) -> str:
        body = {
            **(self.base_body or {}),
            "messages": messages,
        }

        attempt = 0
        while True:
            try:
                completion = self.client.chat.completions.create(**body)
                break
            except (RateLimitError, InternalServerError, APITimeoutError, APIConnectionError) as exc:
                wait_sec = min(5 * (attempt + 1), 60)
                LOGGER.warning(
                    "[AWM] LLM retryable error %s on attempt %s, retrying after %ss",
                    type(exc).__name__,
                    attempt + 1,
                    wait_sec,
                )
                attempt += 1
                if attempt >= self.config.max_retries:
                    raise
                import time
                time.sleep(wait_sec)
            except BadRequestError as exc:
                raise RuntimeError(f"[AWM] Bad request during workflow induction: {exc}") from exc

        choices = getattr(completion, "choices", None) or []
        if not choices:
            return ""

        message = choices[0].message
        content = getattr(message, "content", "") or ""
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
            content = "\n".join(parts)
        return str(content).strip()

    def _bootstrap_rag(self) -> None:
        if not self.rag or not self.config.workflow_storage_path.exists():
            return

        try:
            with self.config.workflow_storage_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    text = str(record.get("text", "") or "").strip()
                    if text:
                        self.rag.insert(key=text, value=text)
        except Exception as exc:
            LOGGER.warning("[AWM] Failed to bootstrap workflows from %s: %s", self.config.workflow_storage_path, exc)

    def _append_workflow_record(self, task: str, text: str) -> None:
        record = {
            "task": task,
            "text": text,
        }
        with self.config.workflow_storage_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _extract_query_from_messages(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        template_titles = [self.template_title]
        question = extract_original_question(messages, where=self.where, template_titles=template_titles)
        if question:
            return str(question).strip()
        return None

    def use_memory(self, task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.rag:
            return list(messages) if messages is not None else []

        query = self._extract_query_from_messages(messages)
        if not query:
            return list(messages) if messages is not None else []

        retrieved = self.rag.retrieve(query=query, top_k=self.rag.top_k)
        if not retrieved:
            return list(messages) if messages is not None else []

        workflow_memory = self.config.prompt_template.format(workflows="\n\n".join(retrieved))
        enhanced = enhance_messages_with_memory(messages, workflow_memory, where=self.where)
        assert_memory_injection_position(enhanced, self.where)
        return enhanced

    def _build_trajectory_text(self, task: str, history: List[Dict[str, Any]]) -> str:
        template_titles = [self.template_title]
        original_question = extract_original_question(history, where=self.where, template_titles=template_titles)

        lines: List[str] = [f"Task: {task}"]
        if original_question:
            lines.append(f"Query: {str(original_question).strip()}")
        lines.append("Trajectory:")

        skip_first_user = True
        for msg in history:
            role, content, msg_dict = extract_message_info(msg)
            if role is None or role == "system":
                continue

            content = str(content or "").strip()

            if role == "user":
                if skip_first_user:
                    skip_first_user = False
                    continue
                if content:
                    lines.append(f"User: {content}")
                continue

            if role == "assistant":
                reasoning = str((msg_dict or {}).get("reasoning_content", "") or "").strip()
                if reasoning:
                    lines.append("Assistant Thought:")
                    lines.append(f"<think>{reasoning}</think>")

                tool_calls = (msg_dict or {}).get("tool_calls", []) or []
                for tool_call in tool_calls:
                    function = tool_call.get("function", {}) or {}
                    tool_name = function.get("name", "unknown_tool")
                    arguments = function.get("arguments", "{}")
                    lines.append(f"Tool: {tool_name}")
                    lines.append(f"Args: {arguments}")

                if content:
                    lines.append(f"Assistant: {content}")
                continue

            if role == "tool" and content:
                lines.append("Observation:")
                lines.append(content)

        return "\n".join(line for line in lines if line is not None).strip()

    def _build_induction_messages(self, trajectory_text: str) -> List[Dict[str, Any]]:
        prompt = "\n\n".join(
            [
                self.instruction_prompt,
                self.one_shot_prompt,
                "Now extract reusable workflows from the following completed task trajectory.",
                trajectory_text,
                "Summary Workflows:",
            ]
        )
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def _split_workflow_blocks(text: str) -> List[str]:
        raw = str(text or "").strip()
        if not raw:
            return []

        blocks = re.split(r"(?=^##\s+)", raw, flags=re.MULTILINE)
        blocks = [block.strip() for block in blocks if block.strip()]
        results: List[str] = []
        for block in blocks:
            normalized = block.lower()
            if not normalized.startswith("## "):
                continue
            if "tool:" not in normalized:
                continue
            results.append(block)
        return results

    def update_memory(self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        status = result.get("status", "")
        reward = result.get("reward", 0)
        is_success = status == "completed"

        if self.config.success_only and not is_success:
            return
        if self.config.reward_bigger_than_zero and reward <= 0:
            return

        trajectory_text = self._build_trajectory_text(task, history)
        if not trajectory_text:
            return

        messages = self._build_induction_messages(trajectory_text)
        response = self._call_llm(messages)
        workflow_blocks = self._split_workflow_blocks(response)
        if not workflow_blocks:
            LOGGER.debug("[AWM] No workflow blocks extracted for task=%s", task)
            return

        for block in workflow_blocks:
            block = str(block or "").strip()
            if not block:
                continue
            if self.rag:
                self.rag.insert(key=block, value=block)
            self._append_workflow_record(task, block)


def load_awm_from_yaml(config_path: str) -> AWM:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict) or "AWM" not in raw or not isinstance(raw["AWM"], dict):
        raise ValueError(
            f"AWM config at {config_path} must use a top-level 'AWM' mapping."
        )

    raw = raw["AWM"]

    rag_cfg = raw.get("workflow_rag", {}) or {}

    config = AWMConfig(
        model_name=str(raw.get("model_name", "gpt-4o-mini")),
        instruction_prompt_path=AWM._resolve_path(
            str(raw.get("instruction_prompt_path", "memory/AWM/prompt/instruction_action.txt"))
        ),
        one_shot_prompt_path=AWM._resolve_path(
            str(raw.get("one_shot_prompt_path", "memory/AWM/prompt/one_shot_action.txt"))
        ),
        workflow_storage_path=AWM._resolve_path(
            str(raw.get("workflow_storage_path", "memory/AWM/workflows.jsonl"))
        ),
        prompt_template=str(raw.get("prompt_template", "Here are some useful workflows:\n\n{workflows}")),
        where=str(raw.get("where", "tail")),
        success_only=bool(raw.get("success_only", True)),
        reward_bigger_than_zero=bool(raw.get("reward_bigger_than_zero", False)),
        workflow_rag_embedding_model=str(rag_cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")),
        workflow_rag_top_k=int(rag_cfg.get("top_k", 5)),
        workflow_rag_order=str(rag_cfg.get("order", "similar_at_top")),
        workflow_rag_seed=int(rag_cfg.get("seed", 42)),
        max_retries=int(raw.get("max_retries", 3)),
    )
    return AWM(config)

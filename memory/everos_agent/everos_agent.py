from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
import yaml

from ..base import MemoryMechanism
from src.utils.message_schema import (
    assert_memory_injection_position,
    enhance_messages_with_memory,
    extract_message_info,
    extract_original_question,
)


LOGGER = logging.getLogger(__name__)


@dataclass
class EverOSAgentConfig:
    api_key: str = ""
    base_url: str = "https://api.evermind.ai"
    user_id: str = "default"
    session_id: Optional[str] = None
    use_session_filter: bool = False
    top_k: int = 10
    search_method: str = "hybrid"
    memory_types: List[str] = field(default_factory=lambda: ["agent_memory"])
    radius: Optional[float] = None
    include_original_data: bool = False
    async_mode: bool = False
    flush_after_add: bool = True
    success_only: bool = True
    reward_bigger_than_zero: bool = False
    prompt_template: str = "Here are relevant agent experiences:\n{memories}"
    where: str = "tail"
    request_timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 2.0
    retry_backoff: float = 2.0
    wait_time: float = 0.0


class EverOSAgentMemory(MemoryMechanism):
    def __init__(self, config: EverOSAgentConfig) -> None:
        self.config = config
        self.template_title = self.config.prompt_template.split("{memories}")[0].strip()
        if not self.config.api_key:
            raise ValueError("EverOS agent memory requires api_key in config")

        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            }
        )

    def _build_url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}{path}"

    def _request_with_retry(
        self,
        method: str,
        path: str,
        json_body: Dict[str, Any],
        expected_statuses: tuple[int, ...],
        purpose: str,
    ) -> requests.Response:
        attempt = 0
        delay = self.config.retry_delay
        while True:
            try:
                response = self._session.request(
                    method=method,
                    url=self._build_url(path),
                    json=json_body,
                    timeout=self.config.request_timeout,
                )
                if response.status_code in expected_statuses:
                    return response
                response.raise_for_status()
                return response
            except requests.RequestException as exc:
                attempt += 1
                if self.config.max_retries >= 0 and attempt > self.config.max_retries:
                    raise RuntimeError(
                        f"[EverOSAgent] {purpose} failed after {attempt} attempts: {exc}"
                    ) from exc
                LOGGER.warning(
                    "[EverOSAgent] %s failed on attempt %s: %s; retrying in %.2fs",
                    purpose,
                    attempt,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay *= self.config.retry_backoff

    def _extract_query(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        template_titles = [self.template_title]
        question = extract_original_question(messages, where=self.config.where, template_titles=template_titles)
        if question:
            return str(question).strip()
        return None

    def _build_search_filters(self) -> Dict[str, Any]:
        filters: Dict[str, Any] = {"user_id": self.config.user_id}
        if self.config.use_session_filter and self.config.session_id:
            filters["session_id"] = self.config.session_id
        return filters

    @staticmethod
    def _format_case(item: Dict[str, Any]) -> Optional[str]:
        task_intent = str(item.get("task_intent") or "").strip()
        approach = str(item.get("approach") or "").strip()
        if not task_intent and not approach:
            return None
        parts: List[str] = []
        if task_intent:
            parts.append(f"Task: {task_intent}")
        if approach:
            parts.append(f"Approach: {approach}")
        score = item.get("score")
        if isinstance(score, (int, float)):
            parts.append(f"Score: {score:.3f}")
        return "- " + " | ".join(parts)

    @staticmethod
    def _format_skill(item: Dict[str, Any]) -> Optional[str]:
        name = str(item.get("name") or "").strip()
        description = str(item.get("description") or "").strip()
        content = str(item.get("content") or "").strip()
        if not name and not description and not content:
            return None
        parts: List[str] = []
        if name:
            parts.append(f"Skill: {name}")
        if description:
            parts.append(f"When to use: {description}")
        if content:
            parts.append(f"Content: {content}")
        score = item.get("score")
        if isinstance(score, (int, float)):
            parts.append(f"Score: {score:.3f}")
        return "- " + " | ".join(parts)

    def _format_search_results(self, data: Dict[str, Any]) -> str:
        lines: List[str] = []
        agent_memory = data.get("agent_memory") or {}
        for item in agent_memory.get("cases", []) or []:
            formatted = self._format_case(item)
            if formatted:
                lines.append(formatted)
        for item in agent_memory.get("skills", []) or []:
            formatted = self._format_skill(item)
            if formatted:
                lines.append(formatted)
        return "\n".join(lines)

    def use_memory(self, task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enhanced = list(messages) if messages is not None else []
        query = self._extract_query(messages)
        if not query:
            return enhanced

        body: Dict[str, Any] = {
            "query": query,
            "filters": self._build_search_filters(),
            "method": self.config.search_method,
            "memory_types": self.config.memory_types,
            "top_k": self.config.top_k,
            "include_original_data": self.config.include_original_data,
        }
        if self.config.radius is not None:
            body["radius"] = self.config.radius

        try:
            response = self._request_with_retry(
                method="POST",
                path="/api/v1/memories/search",
                json_body=body,
                expected_statuses=(200,),
                purpose="search agent memories",
            )
            payload = response.json() if response.content else {}
            data = payload.get("data", {}) if isinstance(payload, dict) else {}
            memory_text = self._format_search_results(data)
            if not memory_text:
                return enhanced
            memory_content = self.config.prompt_template.format(memories=memory_text)
            enhanced = enhance_messages_with_memory(enhanced, memory_content, where=self.config.where)
            assert_memory_injection_position(enhanced, self.config.where)
            return enhanced
        except Exception as exc:
            LOGGER.warning("[EverOSAgent] Search failed for task=%s: %s", task, exc)
            return enhanced

    def _normalize_role(self, role: Any, has_tool_call_id: bool) -> str:
        role_str = str(role or "").strip().lower()
        if role_str == "assistant":
            return "assistant"
        if role_str == "tool" and has_tool_call_id:
            return "tool"
        return "user"

    def _serialize_history(self, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        template_titles = [self.template_title]
        serialized: List[Dict[str, Any]] = []
        base_ts = int(time.time() * 1000)

        for idx, msg in enumerate(history):
            role, content, msg_dict = extract_message_info(msg)
            if role is None:
                continue

            content_value = content
            if role == "user" and content:
                question = extract_original_question([msg], where=self.config.where, template_titles=template_titles)
                if question:
                    content_value = question

            raw = dict(msg_dict) if isinstance(msg_dict, dict) else {"role": role, "content": content_value}
            tool_call_id = raw.get("tool_call_id")
            normalized_role = self._normalize_role(role, bool(tool_call_id))

            item: Dict[str, Any] = {"role": normalized_role, "timestamp": base_ts + idx}
            if normalized_role == "assistant":
                tool_calls = raw.get("tool_calls")
                if isinstance(tool_calls, list) and tool_calls:
                    item["tool_calls"] = tool_calls
                text = str(content_value).strip() if content_value is not None else ""
                item["content"] = text if text else None
            elif normalized_role == "tool":
                text = str(content_value).strip() if content_value is not None else ""
                if not text:
                    continue
                item["content"] = text
                item["tool_call_id"] = str(tool_call_id)
            else:
                text = str(content_value).strip() if content_value is not None else ""
                if not text:
                    continue
                item["content"] = text
            serialized.append(item)

        return serialized

    def _resolve_session_id(self, task: str) -> Optional[str]:
        if self.config.session_id:
            return self.config.session_id
        return task

    def _flush(self, session_id: Optional[str]) -> None:
        body: Dict[str, Any] = {"user_id": self.config.user_id}
        if session_id:
            body["session_id"] = session_id
        self._request_with_retry(
            method="POST",
            path="/api/v1/memories/agent/flush",
            json_body=body,
            expected_statuses=(200,),
            purpose="flush agent memories",
        )

    def update_memory(self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        finish = result.get("finish", False)
        status = result.get("status", "")
        reward = result.get("reward", 0)
        is_success = finish or status == "completed"

        if self.config.success_only and not is_success:
            return
        if self.config.reward_bigger_than_zero and reward <= 0:
            return

        messages = self._serialize_history(history)
        if not messages:
            return

        session_id = self._resolve_session_id(task)
        body: Dict[str, Any] = {
            "user_id": self.config.user_id,
            "messages": messages,
            "async_mode": self.config.async_mode,
        }
        if session_id:
            body["session_id"] = session_id

        try:
            self._request_with_retry(
                method="POST",
                path="/api/v1/memories/agent",
                json_body=body,
                expected_statuses=(200, 202),
                purpose="add agent memories",
            )
            if self.config.flush_after_add:
                self._flush(session_id)
            if self.config.wait_time > 0:
                time.sleep(self.config.wait_time)
        except Exception as exc:
            LOGGER.warning("[EverOSAgent] Update failed for task=%s: %s", task, exc)


def load_everos_agent_from_yaml(config_path: str) -> EverOSAgentMemory:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    raw = cfg.get("everos_agent", {}) or {}
    config = EverOSAgentConfig(
        api_key=str(raw.get("api_key", "")),
        base_url=str(raw.get("base_url", "https://api.evermind.ai")),
        user_id=str(raw.get("user_id", "default")),
        session_id=raw.get("session_id"),
        use_session_filter=bool(raw.get("use_session_filter", False)),
        top_k=int(raw.get("top_k", 10)),
        search_method=str(raw.get("search_method", "hybrid")),
        memory_types=[str(x) for x in (raw.get("memory_types", ["agent_memory"]) or ["agent_memory"])],
        radius=float(raw["radius"]) if raw.get("radius") is not None else None,
        include_original_data=bool(raw.get("include_original_data", False)),
        async_mode=bool(raw.get("async_mode", False)),
        flush_after_add=bool(raw.get("flush_after_add", True)),
        success_only=bool(raw.get("success_only", True)),
        reward_bigger_than_zero=bool(raw.get("reward_bigger_than_zero", False)),
        prompt_template=str(raw.get("prompt_template", "Here are relevant agent experiences:\n{memories}")),
        where=str(raw.get("where", "tail")),
        request_timeout=float(raw.get("request_timeout", 60.0)),
        max_retries=int(raw.get("max_retries", 3)),
        retry_delay=float(raw.get("retry_delay", 2.0)),
        retry_backoff=float(raw.get("retry_backoff", 2.0)),
        wait_time=float(raw.get("wait_time", 0.0)),
    )
    return EverOSAgentMemory(config)

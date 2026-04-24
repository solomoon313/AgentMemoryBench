"""
LLM Agent module.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
LLMAPI_DIR = ROOT_DIR / "configs" / "llmapi"


def _extract_api_key(headers: Dict[str, Any]) -> str:
    auth_value = str(headers.get("Authorization", "") or "").strip()
    if auth_value.lower().startswith("bearer "):
        return auth_value[7:].strip()
    raise ValueError("Authorization header missing Bearer token in llmapi config")


def _normalize_base_url(url: str) -> str:
    normalized = str(url or "").rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    return normalized + "/"


def _message_to_dict(message: Any) -> Dict[str, Any]:
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return dict(message)
    return {}


class SimpleHTTPChatAgent:
    """
    A minimal LLM agent backed by the OpenAI Python SDK.

    - Reads configuration from configs/llmapi/api.yaml + agent.yaml
    - Calls an OpenAI-compatible chat completions endpoint
    - Keeps the existing retry / logging behavior
    """

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.base_url, self.api_key, self.base_body = self._load_agent_config(agent_name)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    @staticmethod
    def _load_agent_config(agent_name: str) -> Tuple[str, str, Dict[str, Any]]:
        agent_cfg_path = LLMAPI_DIR / "agent.yaml"
        api_cfg_path = LLMAPI_DIR / "api.yaml"

        with agent_cfg_path.open("r", encoding="utf-8") as f:
            agents_cfg = yaml.safe_load(f) or {}
        if agent_name not in agents_cfg:
            raise ValueError(f"Agent '{agent_name}' not found in {agent_cfg_path}")

        agent_cfg = agents_cfg[agent_name] or {}

        with api_cfg_path.open("r", encoding="utf-8") as f:
            api_cfg = yaml.safe_load(f) or {}

        base_params = api_cfg.get("parameters", {}) or {}
        agent_params = agent_cfg.get("parameters", {}) or {}

        body = dict(base_params.get("body", {}) or {})
        body.update(agent_params.get("body", {}) or {})

        url = base_params.get("url") or api_cfg.get("parameters", {}).get("url")
        if not url:
            raise ValueError("URL not found in api.yaml / agent.yaml")

        headers = dict(base_params.get("headers", {}) or {})
        headers.update(agent_params.get("headers", {}) or {})

        api_key = _extract_api_key(headers)
        base_url = _normalize_base_url(url)
        return base_url, api_key, body

    def inference(self, history: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        """
        Single-turn call: given a full history (system+user+assistant...), returns one assistant message.
        """
        body: Dict[str, Any] = {
            **(self.base_body or {}),
            "messages": history,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        attempt = 0
        while True:
            try:
                completion = self.client.chat.completions.create(**body)
                break
            except (RateLimitError, InternalServerError) as e:
                wait_sec = min(5 * (attempt + 1), 60)
                logging.warning(
                    "LLM API retryable error %s (attempt %s), retrying after %ss...",
                    type(e).__name__,
                    attempt + 1,
                    wait_sec,
                )
                time.sleep(wait_sec)
                attempt += 1
                continue
            except (APITimeoutError, APIConnectionError) as e:
                wait_sec = min(5 * (attempt + 1), 60)
                logging.warning(
                    "LLM API network/timeout error %s (attempt %s), retrying after %ss...",
                    type(e).__name__,
                    attempt + 1,
                    wait_sec,
                )
                time.sleep(wait_sec)
                attempt += 1
                continue
            except BadRequestError as e:
                raise RuntimeError(
                    f"LLM API bad request: {str(e)}. Request snippet: {json.dumps(body, ensure_ascii=False)[:4000]}"
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"LLM API error {type(e).__name__}: {str(e)}. "
                    f"Request snippet: {json.dumps(body, ensure_ascii=False)[:4000]}"
                ) from e

        choices = getattr(completion, "choices", None) or []
        if not choices:
            raise RuntimeError(f"Empty choices from LLM API: {completion}")

        message = _message_to_dict(choices[0].message)
        if "role" not in message:
            message["role"] = "assistant"
        message["content"] = self._normalize_message_content(message.get("content", ""))
        return message

    @staticmethod
    def _normalize_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text"):
                    text_parts.append(str(item.get("text")))
                    continue
                if item.get("content"):
                    text_parts.append(str(item.get("content")))
            return "\n".join(part for part in text_parts if part).strip()
        if content is None:
            return ""
        return str(content)

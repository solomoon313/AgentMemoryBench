"""
LLM Agent module.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
import yaml
from requests.exceptions import ReadTimeout, Timeout


ROOT_DIR = Path(__file__).resolve().parents[2]
LLMAPI_DIR = ROOT_DIR / "configs" / "llmapi"


class SimpleHTTPChatAgent:
    """
    A minimal LLM agent:
    - Reads HTTP configuration from configs/llmapi/api.yaml + agent.yaml
    - Calls an OpenAI-style chat completions endpoint, supporting tools / tool_choice=auto
    - Simple 429 / 500 retry logic
    """

    def __init__(self, agent_name: str) -> None:
        self.agent_name = agent_name
        self.url, self.headers, self.base_body = self._load_agent_config(agent_name)

    @staticmethod
    def _load_agent_config(agent_name: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Reuses the merge logic from test_client.py:
        - api.yaml provides base parameters (url/headers/body/prompter/return_format)
        - agent.yaml overrides body fields such as model/max_tokens for the specific agent
        """
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

        # Deep-merge body
        body = dict(base_params.get("body", {}) or {})
        body.update(agent_params.get("body", {}) or {})

        url = base_params.get("url") or api_cfg.get("parameters", {}).get("url")
        if not url:
            raise ValueError("URL not found in api.yaml / agent.yaml")

        headers = dict(base_params.get("headers", {}) or {})
        headers.update(agent_params.get("headers", {}) or {})

        return url, headers, body

    def inference(self, history: List[Dict[str, Any]], tools: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        """
        Single-turn call: given a full history (system+user+assistant...), returns one assistant message.
        """
        body: Dict[str, Any] = {
            **(self.base_body or {}),
            "messages": history,
        }
        # Support function calling (tools)
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        # Simple serial retry: 429/500/timeout/network errors retry indefinitely;
        # non-retryable errors raise immediately.
        data: Dict[str, Any] | None = None
        attempt = 0

        while True:
            try:
                # Per-request timeout of 250 seconds to avoid a single sample blocking too long
                resp = requests.post(self.url, headers=self.headers, json=body, timeout=250)
                # Too Many Requests / 500: retry with linear backoff (5s increments, max 60s)
                if resp.status_code in (429, 500):
                    # Linear backoff: 5 * (attempt + 1) seconds (5, 10, 15, ..., 60, 60, ...), max 60s
                    wait_sec = min(5 * (attempt + 1), 60)
                    logging.warning(
                        f"LLM API HTTP {resp.status_code} (attempt {attempt + 1}), "
                        f"retrying after {wait_sec}s (linear backoff, max 60s)..."
                    )
                    time.sleep(wait_sec)
                    attempt += 1
                    continue
                # For other HTTP errors (e.g. 400 Bad Request), raise immediately
                resp.raise_for_status()
                data = resp.json()
                break
            except (ReadTimeout, Timeout) as e:
                # Timeout: retry with linear backoff (5s increments, max 60s)
                wait_sec = min(5 * (attempt + 1), 60)
                logging.warning(
                    f"LLM API timeout (attempt {attempt + 1}), retrying after {wait_sec}s (linear backoff, max 60s)..."
                )
                time.sleep(wait_sec)
                attempt += 1
                continue
            except requests.exceptions.RequestException as e:
                # Other network errors (e.g. connection errors): retry with linear backoff
                wait_sec = min(5 * (attempt + 1), 60)
                logging.warning(
                    f"LLM API network error (attempt {attempt + 1}): {str(e)}, retrying after {wait_sec}s (linear backoff, max 60s)..."
                )
                time.sleep(wait_sec)
                attempt += 1
                continue
            except Exception as e:
                # Other errors (e.g. 400 Bad Request): no retry, raise immediately
                raise RuntimeError(
                    f"LLM API error {getattr(e, 'status_code', 'unknown')}: {str(e)}. "
                    f"Request snippet: {json.dumps(body)[:4000]}"
                ) from e

        # If we exit the loop normally, data is guaranteed non-None (assigned before break)
        # This check is defensive programming and should never actually trigger
        if data is None:
            raise RuntimeError("LLM API call failed: no response data parsed (unexpected state)")
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Empty choices from LLM API: {data}")
        message = choices[0].get("message") or {}
        # Ensure at least role/content fields are present
        if "role" not in message:
            message["role"] = "assistant"
        if "content" not in message:
            message["content"] = ""
        # Preserve reasoning_content if present (returned as-is)
        if "reasoning_content" in message:
            pass
        return message

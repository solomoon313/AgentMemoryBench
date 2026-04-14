"""
Streaming output test script for the evaluate agent:
- Reads configs/llmapi/evaluate_api.yaml and configs/llmapi/evaluate_agent.yaml
- Merges them to obtain the final call config for the specified agent
- Sends a request to the chat/completions endpoint and streams the model reply to the terminal

Usage (from project root):
    python -m src.client.test_evaluate_agent
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import requests
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
LLMAPI_DIR = ROOT_DIR / "configs" / "llmapi"

# System prompt used for all conversations with this demo.
SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "You answer clearly and concisely, can reason over multiple turns, "
    "and you always base your replies only on the conversation history and the user's questions."
)


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simple recursive dict merge: override takes precedence over base.
    """
    result = dict(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = _deep_merge_dict(result[k], v)
        else:
            result[k] = v
    return result


def load_evaluate_agent_config(agent_name: str) -> Dict[str, Any]:
    """
    Load and merge evaluate_api.yaml + evaluate_agent.yaml for the specified agent. Returns:
    {
        "url": ...,
        "headers": {...},
        "body": {...},
    }
    """
    agent_cfg_path = LLMAPI_DIR / "evaluate_agent.yaml"
    api_cfg_path = LLMAPI_DIR / "evaluate_api.yaml"

    with agent_cfg_path.open("r", encoding="utf-8") as f:
        agents_cfg = yaml.safe_load(f) or {}
    if agent_name not in agents_cfg:
        raise ValueError(f"Agent '{agent_name}' not found in {agent_cfg_path}")

    agent_cfg = agents_cfg[agent_name] or {}

    # Read evaluate_api.yaml as the base config
    with api_cfg_path.open("r", encoding="utf-8") as f:
        api_cfg = yaml.safe_load(f) or {}

    base_params = api_cfg.get("parameters", {}) or {}
    agent_params = agent_cfg.get("parameters", {}) or {}

    # Deep-merge parameters (agent overrides api)
    merged_params = _deep_merge_dict(base_params, agent_params)

    url = merged_params.get("url") or api_cfg.get("parameters", {}).get("url")
    if not url:
        raise ValueError("URL not found in evaluate_api.yaml / evaluate_agent.yaml")

    headers = merged_params.get("headers", {}) or api_cfg.get("parameters", {}).get("headers", {})
    body = merged_params.get("body", {}) or api_cfg.get("parameters", {}).get("body", {})

    return {
        "url": url,
        "headers": headers,
        "body": body,
    }


def stream_chat_with_history(
    history: list[dict],
    agent_name: str,
) -> str:
    """
    Send one request based on the given history (including the latest user message),
    stream the assistant's reply to terminal, and return the full text.
    """
    cfg = load_evaluate_agent_config(agent_name)
    url = cfg["url"]
    headers = cfg["headers"]
    base_body = cfg["body"].copy()

    body: Dict[str, Any] = {
        **base_body,
        "messages": history,
        "stream": True,
    }

    full_answer: list[str] = []

    with requests.post(url, headers=headers, json=body, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        # OpenAI-style SSE: each line starts with 'data: '
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="ignore")
            if not line.startswith("data:"):
                continue
            data_str = line[len("data:") :].strip()
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content")
            if content:
                print(content, end="", flush=True)
                full_answer.append(content)

    print()  # newline after streaming
    return "".join(full_answer)


def main() -> None:
    """
    Simple interactive CLI with multi-turn history.
    - First, prompt user to input agent name.
    - Type your messages, press Enter to send.
    - Press Enter on empty line or type 'exit' / 'quit' to stop.
    """
    # Prompt user to select an agent name
    print("Available agents from configs/llmapi/evaluate_agent.yaml:")
    agent_cfg_path = LLMAPI_DIR / "evaluate_agent.yaml"
    with agent_cfg_path.open("r", encoding="utf-8") as f:
        agents_cfg = yaml.safe_load(f) or {}
    for agent_name in agents_cfg.keys():
        print(f"  - {agent_name}")
    print()

    agent_name = input("Enter agent name: ").strip()
    if not agent_name:
        print("Error: Agent name cannot be empty.")
        return

    if agent_name not in agents_cfg:
        print(f"Error: Agent '{agent_name}' not found in {agent_cfg_path}")
        return

    # history follows OpenAI chat format:
    # - role: "system" | "user" | "assistant"
    # - content: text message
    history: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    print(f"\nChat demo (evaluate model: {agent_name}). Press Enter on empty line or type 'exit' to quit.\n")
    print("System:", SYSTEM_PROMPT)
    print()
    while True:
        try:
            user_prompt = input("You: ").strip()
        except EOFError:
            break

        if not user_prompt or user_prompt.lower() in {"exit", "quit"}:
            break

        history.append({"role": "user", "content": user_prompt})

        print("Assistant: ", end="", flush=True)
        answer = stream_chat_with_history(history, agent_name=agent_name)

        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()


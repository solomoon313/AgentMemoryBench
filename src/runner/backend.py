"""
Backend client module.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import requests

from src.client.scheduler import SampleIndex


class BackendClient:
    """
    Minimal backend client wrapping /start_sample and /list_workers.
    Additional endpoints such as /interact and /calculate_overall can be added here.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def list_workers(self) -> Dict[str, Any]:
        url = f"{self.base_url}/list_workers"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_indices(self, task_name: str) -> List[SampleIndex]:
        url = f"{self.base_url}/get_indices"
        resp = requests.get(url, params={"name": task_name}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected indices response for task {task_name}: {data}")
        return list(data)

    def start_sample(self, task_name: str, index: int) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Call /start_sample and return (session_id, messages, tools).
        The backend returns a TaskOutput dict containing messages/tools.
        """
        url = f"{self.base_url}/start_sample"
        payload = {"name": task_name, "index": index}
        resp = requests.post(url, json=payload, timeout=120)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            response_text = (resp.text or "").strip()
            if response_text:
                raise requests.HTTPError(
                    f"{e}. Response body: {response_text}",
                    response=resp,
                    request=resp.request,
                ) from e
            raise
        # session_id is returned in response headers
        session_id_header = resp.headers.get("session_id")
        if session_id_header is None:
            raise RuntimeError("Backend /start_sample did not return session_id in headers")
        try:
            session_id = int(session_id_header)
        except ValueError:
            raise RuntimeError(f"Invalid session_id header: {session_id_header}")

        data = resp.json()
        messages = data.get("messages", []) or []
        tools = data.get("tools", []) or []
        return session_id, messages, tools

    def interact(self, session_id: int, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Call /interact to send the agent's reply to the backend environment controller.

        Following the AgentBench client convention:
        - session_id is passed in an HTTP header
        - JSON body contains only {"messages": [...]}
        - Return value is the env_result (containing finish/status/reward/messages etc.)
        """
        url = f"{self.base_url}/interact"
        headers = {"session_id": str(session_id)}
        payload = {"messages": messages}
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        resp.raise_for_status()
        data = resp.json()
        return data

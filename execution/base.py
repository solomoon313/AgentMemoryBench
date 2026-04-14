from __future__ import annotations

from typing import Protocol, List, Dict, Any, Tuple


class ExecutionEngine(Protocol):
    """
    Abstract interface for execution engines.

    Responsible for, given already-constructed messages + tools:
    - Selecting the appropriate agent (single / multi-agent)
    - Calling the LLM
    - Conducting multi-turn /interact exchanges with the backend
    - Returning the complete history and final result (reward / status / metric, etc.)
    """

    def run_sample(
        self,
        task: str,
        index: int,
        session_id: int,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        agent_pool: Any,
        backend_client: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute the complete interaction flow for one sample.

        Returns:
        - history: full conversation trajectory in OpenAI Chat format (system/user/assistant/tool/...)
        - result: final result returned or aggregated by the backend (reward/status/metric, etc.)
        """



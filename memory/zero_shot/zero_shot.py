from __future__ import annotations

from typing import List, Dict, Any

import yaml

from ..base import MemoryMechanism


class ZeroShotMemory(MemoryMechanism):
    """
    Zero-shot memory mechanism: does not leverage any historical experience
    and does not update any memory store.

    - use_memory: passes through the backend messages unchanged
    - update_memory: no-op

    This class exists so that all memory mechanisms share the same interface,
    making it easy to switch between them via configuration in the runner.
    """

    def use_memory(self, task: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Return a shallow copy to prevent callers from mutating the original list
        return list(messages) if messages is not None else []

    def update_memory(self, task: str, history: List[Dict[str, Any]], result: Dict[str, Any]) -> None:
        # Zero-shot records no experience; intentionally left empty
        return


def load_zero_shot_from_yaml(config_path: str) -> ZeroShotMemory:
    """
    Load configuration from memory/zero_shot/zero_shot.yaml and construct a ZeroShotMemory.

    The YAML currently only contains descriptive metadata, so this function simply
    validates that the file exists and is well-formed, then returns a ZeroShotMemory
    instance. The structure is kept for future extensibility (e.g., per-task static prompts).
    """
    with open(config_path, "r", encoding="utf-8") as f: #using "utf-8" to prevent there are some CHINESE in yaml file
        _ = yaml.safe_load(f) or {}  # safe_load returns None for empty files; fall back to {}
    return ZeroShotMemory()
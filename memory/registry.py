"""
Memory Mechanism Registry - centralized management for registering and loading all memory mechanisms

Naming convention:
- Use snake_case in configuration (e.g. stream_icl, awm_pro, mems)
- The registry maps these names to their actual classes and loader functions
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any

from memory.base import MemoryMechanism


# Registry: name -> (loader_function, default_config_path)
_MEMORY_REGISTRY: Dict[str, tuple[Callable[[str], MemoryMechanism], str]] = {}


def register_memory(
    name: str,
    loader_func: Callable[[str], MemoryMechanism],
    default_config_path: str,
) -> None:
    """
    Register a memory mechanism.

    Args:
        name: mechanism name in snake_case (e.g. stream_icl, awm_pro)
        loader_func: loader function that accepts a config_path and returns a MemoryMechanism instance
        default_config_path: default config file path relative to the project root
    """
    _MEMORY_REGISTRY[name] = (loader_func, default_config_path)


def get_memory_loader(name: str) -> tuple[Callable[[str], MemoryMechanism], str]:
    """
    Get the loader function and default config path for a memory mechanism.

    Args:
        name: mechanism name

    Returns:
        (loader_func, default_config_path)

    Raises:
        ValueError: if the mechanism is not registered
    """
    if name not in _MEMORY_REGISTRY:
        available = ", ".join(sorted(_MEMORY_REGISTRY.keys()))
        raise ValueError(
            f"Memory mechanism '{name}' not registered. "
            f"Available mechanisms: {available}"
        )
    return _MEMORY_REGISTRY[name]


def list_available_memories() -> list[str]:
    """Return names of all registered memory mechanisms."""
    return sorted(_MEMORY_REGISTRY.keys())


# ===== Register all memory mechanisms =====

def _register_all_memories():
    """Register all built-in memory mechanisms."""

    # zero_shot
    from memory.zero_shot.zero_shot import load_zero_shot_from_yaml
    register_memory(
        name="zero_shot",
        loader_func=load_zero_shot_from_yaml,
        default_config_path="memory/zero_shot/zero_shot.yaml",
    )

    # stream_icl (snake_case)
    from memory.streamICL.streamICL import load_stream_icl_from_yaml
    register_memory(
        name="stream_icl",
        loader_func=load_stream_icl_from_yaml,
        default_config_path="memory/streamICL/streamICL.yaml",
    )

    # mem0
    from memory.mem0.mem0 import load_mem0_from_yaml
    register_memory(
        name="mem0",
        loader_func=load_mem0_from_yaml,
        default_config_path="memory/mem0/mem0.yaml",
    )

    # mems (lowercase)
    from memory.MEMs import load_mems_from_yaml
    register_memory(
        name="mems",
        loader_func=load_mems_from_yaml,
        default_config_path="memory/MEMs/MEMs.yaml",
    )

    # awm_pro (snake_case)
    from memory.awmPro.awmPro import load_awmpro_from_yaml
    register_memory(
        name="awm_pro",
        loader_func=load_awmpro_from_yaml,
        default_config_path="memory/awmPro/awmPro.yaml",
    )


# Auto-register all memory mechanisms on import
_register_all_memories()

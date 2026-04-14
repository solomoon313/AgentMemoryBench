"""
Configuration loading module.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
ASSIGNMENT_PATH = ROOT_DIR / "configs" / "assignment" / "default.yaml"


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    tasks: List[Dict[str, Any]]
    memory_mechanism: Dict[str, Any]
    execution_method: Dict[str, Any]
    experiment: Dict[str, Any]


def load_experiment_config() -> ExperimentConfig:
    """Load experiment configuration from default.yaml."""
    with ASSIGNMENT_PATH.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return ExperimentConfig(
        tasks=raw.get("tasks", []) or [],
        memory_mechanism=raw.get("memory_mechanism", {}) or {},
        execution_method=raw.get("execution_method", {}) or {},
        experiment=raw.get("experiment", {}) or {},
    )

"""
Execution engines for the lifelong-learning benchmark.

Currently implemented:
- single_agent: a single LLM agent that runs through an entire sample
"""

from .base import ExecutionEngine  # noqa: F401
# single_agent is a sub-package; implementation lives in execution/single_agent/single_agent.py
from .single_agent.single_agent import SingleAgentExecutionEngine  # noqa: F401



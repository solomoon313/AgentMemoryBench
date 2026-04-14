"""
Memory mechanisms for the lifelong-learning benchmark.

Currently only zero_shot is implemented as a placeholder for the unified interface.
The following will be added to this directory in future iterations:
- stream_icl
- previous_sample_utilization
- agent_workflow_memory
- mem0
- context_compression
"""

from .base import MemoryMechanism  # noqa: F401
# zero_shot is a sub-package; implementation lives in memory/zero_shot/zero_shot.py
from .zero_shot.zero_shot import ZeroShotMemory  # noqa: F401
# context_compression is a sub-package; implementation lives in memory/context_compression/context_compression.py
# from .context_compression.context_compression import ConversationCompressionMemory  # noqa: F401  # TODO: not implemented



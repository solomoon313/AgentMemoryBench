from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


TaskName = str
SampleIndex = int
Schedule = List[Tuple[TaskName, SampleIndex]]


@dataclass
class ScheduleConfig:
    cross_task: bool
    shuffle: bool
    seed: int | None = None


def build_schedule(
    task_to_indices: Dict[TaskName, Sequence[SampleIndex]],
    config: ScheduleConfig,
) -> Schedule:
    """
    Build a unified (task_name, sample_index) schedule according to the global
    lifelong-learning settings (cross_task, shuffle, interval, ...).
    """
    if not task_to_indices:
        return []

    if config.cross_task and not config.shuffle:
        # cross_task=True with shuffle=False is not supported
        raise ValueError(
            "cross_task=True and shuffle=False is not supported. "
            "Please use either cross_task=False or shuffle=True."
        )

    if not config.cross_task and not config.shuffle:
        # Case A: per-task in-order
        return _schedule_sequential(task_to_indices)

    if not config.cross_task and config.shuffle:
        # Case B: per-task shuffle, no cross-task mixing
        return _schedule_sequential_shuffled(task_to_indices, config.seed)

    # Case C: cross_task=True and shuffle=True -> global shuffle of all samples
    return _schedule_global_shuffle(task_to_indices, config.seed)


def _schedule_sequential(task_to_indices: Dict[TaskName, Sequence[SampleIndex]]) -> Schedule:
    schedule: Schedule = []
    for task, indices in task_to_indices.items():
        for idx in indices:
            schedule.append((task, idx))
    return schedule


def _schedule_sequential_shuffled(
    task_to_indices: Dict[TaskName, Sequence[SampleIndex]],
    seed: int | None,
) -> Schedule:
    import random

    rng = random.Random(seed)
    schedule: Schedule = []
    for task, indices in task_to_indices.items():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        for idx in shuffled:
            schedule.append((task, idx))
    return schedule


def _schedule_global_shuffle(
    task_to_indices: Dict[TaskName, Sequence[SampleIndex]],
    seed: int | None,
) -> Schedule:
    import random

    all_pairs: Schedule = []
    for task, indices in task_to_indices.items():
        for idx in indices:
            all_pairs.append((task, idx))

    rng = random.Random(seed)
    rng.shuffle(all_pairs)
    return all_pairs



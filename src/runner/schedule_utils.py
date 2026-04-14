"""
Schedule utilities module - handles task scheduling and schedule construction.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from src.client.scheduler import ScheduleConfig, build_schedule, TaskName, SampleIndex, Schedule
from src.runner.backend import BackendClient
from src.runner.config import ExperimentConfig, ROOT_DIR
from src.server.tasks.locomo.task import LocomoBaseTask


# Special marker: indicates a session content injection
SESSION_INJECTION_MARKER = "__SESSION_INJECTION__"
# Special marker: indicates a replay-mode test sample
REPLAY_TEST_MARKER = "__REPLAY_TEST__"
# Special marker: indicates a repair-mode group marker
REPAIR_GROUP_MARKER = "__REPAIR_GROUP__"

def load_task_instance(task_name: str, exp_cfg: ExperimentConfig):
    """Load the task instance for the given task_name (for special locomo task handling)."""
    # Find task config
    task_cfg = None
    for t in exp_cfg.tasks:
        if t.get("name") == task_name:
            task_cfg = t
            break

    if not task_cfg:
        print(f"[load_task_instance] Task config not found for {task_name}")
        return None

    config_path = task_cfg.get("config_path")
    if not config_path:
        print(f"[load_task_instance] config_path not found for {task_name}")
        return None

    # Load YAML config
    config_path = ROOT_DIR / config_path
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            task_yaml = yaml.safe_load(f) or {}
    except Exception as e:
        print(f"[load_task_instance] Failed to load YAML from {config_path}: {e}")
        return None

    # Get task-specific config (if any)
    # Use default config as the base
    default_cfg = task_yaml.get("default", {})
    task_specific_cfg = task_yaml.get(task_name, {})

    # Merge configs: default as base, task_specific overrides
    merged_cfg = default_cfg.copy() if default_cfg else {}
    if task_specific_cfg:
        # Merge parameters (if present)
        if "parameters" in task_specific_cfg:
            merged_params = merged_cfg.get("parameters", {}).copy() if merged_cfg.get("parameters") else {}
            merged_params.update(task_specific_cfg.get("parameters", {}))
            merged_cfg["parameters"] = merged_params
        # Override module if task_specific has one
        if "module" in task_specific_cfg:
            merged_cfg["module"] = task_specific_cfg["module"]

    if not merged_cfg:
        print(f"[load_task_instance] No config found for {task_name} in {config_path}")
        return None

    module_path = merged_cfg.get("module", "")
    parameters = merged_cfg.get("parameters", {}) or {}

    print(f"[load_task_instance] module_path={module_path}, parameters={parameters}")

    # Dynamically import and instantiate
    try:
        # Support all locomo tasks (locomo-0 through locomo-9)
        task_classes = {
            f"locomo-{i}": LocomoBaseTask for i in range(10)
        }

        for task_name_key, task_class in task_classes.items():
            if task_name_key in module_path or "LocomoBaseTask" in module_path:
                return task_class(**parameters)

        print(f"[load_task_instance] Unknown module_path: {module_path}")
        return None
    except Exception as e:
        print(f"[load_task_instance] Failed to instantiate task: {e}")
        import traceback
        traceback.print_exc()
        return None


def is_locomo_task(task_name: str) -> bool:
    """Return True if the task is a locomo task."""
    return task_name in tuple(f"locomo-{i}" for i in range(10))


def build_locomo_session_schedule(
    locomo_task_name: TaskName,
    locomo_task_instance: Any,
    shuffle_enabled: bool,
    seed: int | None,
) -> Schedule:
    """
    Build a session-order schedule for the locomo task (used in online mode, cross_task=False).

    Processes sessions in order: inject session1 content, then process session1 QAs
    (optionally shuffled), then inject session2, process session2 QAs, and so on.

    Args:
        locomo_task_name: locomo task name
        locomo_task_instance: locomo task instance
        shuffle_enabled: whether to shuffle (affects only intra-session QA order)
        seed: random seed for shuffling

    Returns:
        schedule sequence where session injection uses special marker (SESSION_INJECTION_MARKER, session_id)
    """
    import random as rnd

    rng = rnd.Random(seed)

    schedule: Schedule = []
    session_ids = locomo_task_instance.session_ids
    print(f"[Locomo Session Schedule] Processing {len(session_ids)} sessions: {session_ids}")

    for session_id in session_ids:
        # 1. Insert session injection marker
        schedule.append((SESSION_INJECTION_MARKER, session_id))

        # 2. Get all QA indices for this session
        qa_indices = locomo_task_instance.get_qa_indices_for_session(session_id)

        # 3. If shuffle=True, shuffle QA order within this session
        if shuffle_enabled:
            qa_list = list(qa_indices)
            rng.shuffle(qa_list)
            schedule.extend([(locomo_task_name, qa_idx) for qa_idx in qa_list])
            print(f"  -> Session {session_id}: {len(qa_list)} QAs (shuffled)")
        else:
            schedule.extend([(locomo_task_name, qa_idx) for qa_idx in qa_indices])
            print(f"  -> Session {session_id}: {len(qa_indices)} QAs (original order)")

    print(f"[Locomo Session Schedule] Total schedule length: {len(schedule)}")
    return schedule


def build_transfer_schedule(
    task_to_indices: Dict[TaskName, List[SampleIndex]],
    transfer_task: TaskName,
    transfer_after_task: TaskName,
    shuffle_enabled: bool,
    seed: int | None,
) -> Schedule:
    """
    Build a transfer-mode schedule: run all transfer_task samples first (training),
    then all transfer_after_task samples (testing).

    Args:
        task_to_indices: mapping from task to sample indices
        transfer_task: training task name (update+enhance)
        transfer_after_task: test task name (enhance only)
        shuffle_enabled: whether to shuffle
        seed: random seed for shuffling

    Returns:
        schedule: transfer_task samples first, then transfer_after_task samples
    """
    import random as rnd

    schedule: Schedule = []

    # 1. Process transfer_task (training task) first
    if transfer_task not in task_to_indices:
        raise ValueError(f"transfer_task '{transfer_task}' not found in task_to_indices")
    transfer_indices = list(task_to_indices[transfer_task])

    if shuffle_enabled:
        rng = rnd.Random(seed)
        rng.shuffle(transfer_indices)
        print(f"[Transfer Schedule] Shuffled {len(transfer_indices)} samples for transfer_task={transfer_task}")
    else:
        print(f"[Transfer Schedule] {len(transfer_indices)} samples for transfer_task={transfer_task} (no shuffle)")

    schedule.extend([(transfer_task, idx) for idx in transfer_indices])

    # 2. Process transfer_after_task (test task) second
    if transfer_after_task not in task_to_indices:
        raise ValueError(f"transfer_after_task '{transfer_after_task}' not found in task_to_indices")
    transfer_after_indices = list(task_to_indices[transfer_after_task])

    if shuffle_enabled:
        # Use the same seed but create a new RNG so each task shuffles independently
        rng = rnd.Random(seed)
        rng.shuffle(transfer_after_indices)
        print(f"[Transfer Schedule] Shuffled {len(transfer_after_indices)} samples for transfer_after_task={transfer_after_task}")
    else:
        print(f"[Transfer Schedule] {len(transfer_after_indices)} samples for transfer_after_task={transfer_after_task} (no shuffle)")

    schedule.extend([(transfer_after_task, idx) for idx in transfer_after_indices])

    print(f"[Transfer Schedule] Total schedule length: {len(schedule)} (train={len(transfer_indices)}, test={len(transfer_after_indices)})")
    return schedule


def build_replay_schedule(
    task_to_indices: Dict[TaskName, List[SampleIndex]],
    replay_m: int,
    replay_n: int,
    replay_seed: int,
    shuffle_enabled: bool,
    seed: int | None,
) -> Tuple[Schedule, Dict[int, Dict[str, List[SampleIndex]]]]:
    """
    Build a replay-mode schedule: after every replay_m training samples, randomly sample
    replay_n samples from all learned samples for testing.

    Args:
        task_to_indices: task-to-indices mapping (must have exactly one task)
        replay_m: number of training samples per replay interval
        replay_n: number of samples to test per replay
        replay_seed: random seed for test sample sampling
        shuffle_enabled: whether to shuffle training samples
        seed: random seed for training sample shuffling

    Returns:
        (schedule, replay_info):
        - schedule: alternating train and test samples
        - replay_info: {replay_id: {"train": [...], "test": [...]}}
    """
    import random as rnd

    if len(task_to_indices) != 1:
        raise ValueError(f"replay mode requires exactly 1 task, but got {len(task_to_indices)} tasks")

    task_name = list(task_to_indices.keys())[0]
    all_indices = list(task_to_indices[task_name])

    # 1. Prepare training samples (shuffle if enabled)
    train_indices = list(all_indices)
    if shuffle_enabled:
        rng = rnd.Random(seed)
        rng.shuffle(train_indices)
        print(f"[Replay Schedule] Shuffled {len(train_indices)} training samples")
    else:
        print(f"[Replay Schedule] {len(train_indices)} training samples (no shuffle)")

    # 2. Build schedule: after every replay_m train samples, sample replay_n test samples
    schedule: Schedule = []
    test_rng = rnd.Random(replay_seed)  # Use replay_seed for test sample sampling

    learned_samples: List[SampleIndex] = []  # All samples learned so far
    replay_info: Dict[int, Dict[str, List[SampleIndex]]] = {}  # Info per replay batch

    replay_id = 1
    for i in range(0, len(train_indices), replay_m):
        # Add current batch of training samples
        batch = train_indices[i:i + replay_m]
        schedule.extend([(task_name, idx) for idx in batch])
        learned_samples.extend(batch)

        # Sample replay_n test samples from all learned samples
        if len(learned_samples) > 0:
            # If fewer than replay_n samples learned, use all of them
            n_samples = min(replay_n, len(learned_samples))
            test_samples = test_rng.sample(learned_samples, n_samples)
            # Use special marker to identify test samples
            schedule.extend([(REPLAY_TEST_MARKER, idx) for idx in test_samples])

            # Record this replay batch's info
            replay_info[replay_id] = {
                "train": learned_samples.copy(),  # All samples learned up to this replay
                "test": test_samples.copy()        # Samples tested in this replay
            }

            print(f"[Replay Schedule] Replay {replay_id}: {len(batch)} new train, {len(test_samples)} test (from {len(learned_samples)} learned)")
            replay_id += 1

    print(f"[Replay Schedule] Total schedule length: {len(schedule)} (train={len(train_indices)}, {len(replay_info)} replays)")
    return schedule, replay_info


def build_replay_schedule_for_locomo(
    task_name: TaskName,
    locomo_task_instance: Any,
    shuffle_enabled: bool,
    seed: int | None,
) -> Tuple[Schedule, Dict[int, Dict[str, List[SampleIndex]]]]:
    """
    Build a replay-mode schedule for locomo tasks: partition by session,
    each session is one replay batch.

    For locomo tasks, replay mode partitions by session:
    - Replay 1: all QAs from session 1 (train), then all QAs from session 1 (test)
    - Replay 2: all QAs from sessions 1+2 (train), then sample from sessions 1+2 (test)
    - ...

    Args:
        task_name: locomo task name
        locomo_task_instance: locomo task instance
        shuffle_enabled: whether to shuffle QAs within each session
        seed: random seed for shuffling

    Returns:
        (schedule, replay_info):
        - schedule: session injection markers + QA samples
        - replay_info: {replay_id: {"train": [...], "test": [...]}}
    """
    import random as rnd

    rng = rnd.Random(seed) if shuffle_enabled else None

    schedule: Schedule = []
    replay_info: Dict[int, Dict[str, List[SampleIndex]]] = {}

    session_ids = locomo_task_instance.session_ids
    learned_samples: List[SampleIndex] = []  # All QA indices learned so far

    print(f"[Locomo Replay Schedule] Processing {len(session_ids)} sessions: {session_ids}")

    replay_id = 1
    for session_id in session_ids:
        # 1. Inject current session's content into memory
        schedule.append((SESSION_INJECTION_MARKER, session_id))

        # 2. Get all QA indices for this session
        session_qa_indices = locomo_task_instance.get_qa_indices_for_session(session_id)

        # 3. If shuffle=True, shuffle QA order within this session
        if shuffle_enabled and rng:
            qa_list = list(session_qa_indices)
            rng.shuffle(qa_list)
            session_qa_indices = qa_list

        # 4. Add this session's QAs as training samples
        schedule.extend([(task_name, qa_idx) for qa_idx in session_qa_indices])
        learned_samples.extend(session_qa_indices)

        # 5. Sample test samples from all learned QAs
        # For locomo, m/n parameters are ignored; use all QAs from the current session as test
        test_samples = session_qa_indices.copy()
        schedule.extend([(REPLAY_TEST_MARKER, qa_idx) for qa_idx in test_samples])

        # 6. Record this replay batch's info
        replay_info[replay_id] = {
            "train": learned_samples.copy(),  # All QAs learned up to this session
            "test": test_samples.copy()        # All QAs from the current session
        }

        print(f"[Locomo Replay Schedule] Replay {replay_id} (Session {session_id}): {len(session_qa_indices)} train, {len(test_samples)} test (total learned: {len(learned_samples)})")
        replay_id += 1

    print(f"[Locomo Replay Schedule] Total schedule length: {len(schedule)} ({len(replay_info)} replays)")
    return schedule, replay_info


def build_mixed_schedule(
    system_memory_tasks: Dict[TaskName, List[SampleIndex]],
    locomo_task_name: TaskName,
    locomo_task_instance: Any,
    shuffle_enabled: bool,
    seed: int | None,
) -> Schedule:
    """
    Build a mixed schedule: shuffle system memory tasks first, then interleave
    locomo tasks in session order.

    Args:
        system_memory_tasks: dict of system memory task samples
        locomo_task_name: locomo task name
        locomo_task_instance: locomo task instance
        shuffle_enabled: whether to shuffle
        seed: random seed for shuffling

    Returns:
        mixed schedule where session injection uses special marker (SESSION_INJECTION_MARKER, session_id)
    """
    import random as rnd

    rng = rnd.Random(seed)

    # 1. Shuffle all system memory task samples
    system_memory_schedule: Schedule = []
    for task_name, indices in system_memory_tasks.items():
        for idx in indices:
            system_memory_schedule.append((task_name, idx))

    if shuffle_enabled:
        rng.shuffle(system_memory_schedule)

    print(f"[Mixed Schedule] Shuffled {len(system_memory_schedule)} system memory samples")

    # 2. Process locomo task sessions in order
    if locomo_task_instance is None:
        return system_memory_schedule

    session_ids = locomo_task_instance.session_ids
    print(f"[Mixed Schedule] Processing {len(session_ids)} locomo sessions: {session_ids}")

    # 3. Prepare QA lists for each session
    session_qa_map: Dict[int, List[SampleIndex]] = {}
    for session_id in session_ids:
        qa_indices = locomo_task_instance.get_qa_indices_for_session(session_id)
        if shuffle_enabled:
            qa_list = list(qa_indices)
            rng.shuffle(qa_list)
            session_qa_map[session_id] = qa_list
        else:
            session_qa_map[session_id] = list(qa_indices)
        print(f"  -> Session {session_id}: {len(session_qa_map[session_id])} QAs")

    # 4. Build mixed schedule
    # Strategy: sessions are processed in order; each session's injection and all its QAs
    # must complete before the next session starts, with system samples interleaved
    mixed_schedule: Schedule = []
    system_idx = 0  # Current position in system_memory_schedule

    for session_idx, session_id in enumerate(session_ids):
        qa_list = session_qa_map[session_id]
        if not qa_list:
            continue

        # Calculate available system samples for this session
        remaining_system = len(system_memory_schedule) - system_idx
        if remaining_system <= 0:
            # System samples exhausted: insert this session's injection and all QAs directly
            mixed_schedule.append((SESSION_INJECTION_MARKER, session_id))
            for qa_idx in qa_list:
                mixed_schedule.append((locomo_task_name, qa_idx))
            print(f"  -> Session {session_id}: injection + {len(qa_list)} QAs (no system samples remaining)")
            continue

        # Calculate how many system samples to allocate to this session
        # Reserve samples for remaining sessions (if this is not the last session)
        is_last_session = (session_idx == len(session_ids) - 1)
        if is_last_session:
            # Last session: use all remaining samples
            session_db_count = remaining_system
        else:
            # Not last session: distribute remaining samples evenly
            remaining_sessions = len(session_ids) - session_idx
            session_db_count = remaining_system // remaining_sessions
            session_db_count = max(1, session_db_count)  # Use at least 1 sample

        # Record this session's system sample range
        session_start_idx = system_idx
        session_end_idx = min(system_idx + session_db_count, len(system_memory_schedule))

        # 4.1 Choose a position within this session's range to insert the session injection
        if session_db_count > 1:
            # Pick a random injection position within the first 50% of the range
            injection_offset = rng.randint(0, max(1, session_db_count // 2))
            injection_pos = session_start_idx + injection_offset
        else:
            injection_pos = session_start_idx

        # Insert system samples up to the injection position
        while system_idx < injection_pos and system_idx < session_end_idx:
            mixed_schedule.append(system_memory_schedule[system_idx])
            system_idx += 1

        # 4.2 Insert session injection marker
        mixed_schedule.append((SESSION_INJECTION_MARKER, session_id))
        print(f"  -> Inserted session {session_id} injection at position {len(mixed_schedule) - 1}")

        # 4.3 Distribute QAs within the remaining system samples for this session
        session_remaining_db = session_end_idx - system_idx

        if session_remaining_db > 0 and len(qa_list) > 0:
            # Calculate intervals between QAs
            if len(qa_list) == 1:
                intervals = [session_remaining_db]
            else:
                # Divide remaining system samples into len(qa_list) segments
                base_interval = session_remaining_db // (len(qa_list) + 1)
                intervals = []
                remaining = session_remaining_db
                for i in range(len(qa_list)):
                    if i == len(qa_list) - 1:
                        # Last QA: use all remaining positions
                        intervals.append(remaining)
                    else:
                        # Random interval around base_interval
                        interval = base_interval + rng.randint(-base_interval // 2, base_interval // 2)
                        interval = max(1, min(interval, remaining - (len(qa_list) - i - 1)))
                        intervals.append(interval)
                        remaining -= interval
        else:
            # No remaining system samples: insert all QAs directly
            intervals = [0] * len(qa_list)

        # 4.4 Insert QAs at the calculated intervals (within this session's system sample range)
        for qa_idx, interval in zip(qa_list, intervals):
            # Insert system samples first
            for _ in range(interval):
                if system_idx < session_end_idx and system_idx < len(system_memory_schedule):
                    mixed_schedule.append(system_memory_schedule[system_idx])
                    system_idx += 1
            # Then insert QA
            mixed_schedule.append((locomo_task_name, qa_idx))

        print(f"  -> Inserted {len(qa_list)} QAs for session {session_id} (used {session_db_count} system samples, range: {session_start_idx}-{session_end_idx})")

    # 5. Append any remaining system memory samples
    while system_idx < len(system_memory_schedule):
        mixed_schedule.append(system_memory_schedule[system_idx])
        system_idx += 1

    print(f"[Mixed Schedule] Final schedule length: {len(mixed_schedule)}")
    print(f"  -> System memory samples: {len(system_memory_schedule)}")
    print(f"  -> Locomo session injections: {len(session_ids)}")
    print(f"  -> Locomo QAs: {sum(len(session_qa_map[sid]) for sid in session_ids)}")

    return mixed_schedule


def build_offline_locomo_schedule(
    locomo_task_name: TaskName,
    locomo_task_instance: Any,
    shuffle_enabled: bool,
    seed: int | None,
) -> Schedule:
    """
    Build an offline-mode locomo task schedule: inject all sessions at once,
    then process all QAs.

    Offline mode characteristics:
    - Inject all session content at the beginning (using SESSION_INJECTION_MARKER)
    - Then process all QAs in order (or shuffled)
    - This ensures all session information is in memory before train/test splitting

    Args:
        locomo_task_name: locomo task name
        locomo_task_instance: locomo task instance
        shuffle_enabled: whether to shuffle QAs
        seed: random seed for shuffling

    Returns:
        schedule: all session injections first, then all QAs
    """
    import random as rnd

    schedule: Schedule = []
    session_ids = locomo_task_instance.session_ids

    print(f"[Offline Locomo Schedule] Processing {len(session_ids)} sessions: {session_ids}")

    # 1. Inject all sessions at the beginning
    for session_id in session_ids:
        schedule.append((SESSION_INJECTION_MARKER, session_id))
    print(f"[Offline Locomo Schedule] Added {len(session_ids)} session injection markers at the beginning")

    # 2. Collect all QA indices
    all_qa_indices: List[SampleIndex] = []
    for session_id in session_ids:
        qa_indices = locomo_task_instance.get_qa_indices_for_session(session_id)
        all_qa_indices.extend(qa_indices)

    # 3. Shuffle all QAs if enabled
    if shuffle_enabled:
        rng = rnd.Random(seed)
        rng.shuffle(all_qa_indices)
        print(f"[Offline Locomo Schedule] Shuffled {len(all_qa_indices)} QAs")
    else:
        print(f"[Offline Locomo Schedule] {len(all_qa_indices)} QAs (original order)")

    # 4. Append all QAs to the schedule
    for qa_idx in all_qa_indices:
        schedule.append((locomo_task_name, qa_idx))

    print(f"[Offline Locomo Schedule] Total schedule length: {len(schedule)} ({len(session_ids)} injections + {len(all_qa_indices)} QAs)")
    return schedule


def build_repair_schedule(
    task_to_indices: Dict[TaskName, List[SampleIndex]],
    repair_m: int,
    repair_n: int,
    repair_seed: int,
    shuffle_enabled: bool,
    seed: int | None,
) -> Tuple[Schedule, Dict[int, Dict[str, Any]]]:
    """
    Build a repair-mode schedule: tests the memory system's ability to handle knowledge conflicts.

    Repair mode workflow:
    1. Divide all samples into groups of size repair_m (repair1, repair2, ...)
    2. Within each group, randomly select repair_n samples for reward reversal
    3. Each group executes 4 phases:
       - wrongJudgeFull: learn all m samples (with reversed rewards)
       - wrongJudgeStandard: learn only n reversed samples (with reversed rewards)
       - wrongJudgeTestFull: test all m samples (with correct rewards)
       - wrongJudgeTestStandard: test n reversed samples (with correct rewards)
       - rightJudgeFull: re-learn all m samples (with correct rewards)
       - rightJudgeStandard: learn only n reversed samples (with correct rewards)
       - rightJudgeTestFull: test all m samples (with correct rewards)
       - rightJudgeTestStandard: test n reversed samples (with correct rewards)

    Args:
        task_to_indices: task-to-indices mapping (must have exactly one task)
        repair_m: samples per group
        repair_n: number of samples to reverse per group
        repair_seed: random seed for selecting reversed samples
        shuffle_enabled: whether to shuffle all samples before grouping
        seed: random seed for shuffling

    Returns:
        (schedule, repair_info):
        - schedule: repair group markers and samples
        - repair_info: {repair_id: {"all_samples": [...], "reversed_samples": [...]}}
    """
    import random as rnd

    if len(task_to_indices) != 1:
        raise ValueError(f"repair mode requires exactly 1 task, but got {len(task_to_indices)} tasks")

    task_name = list(task_to_indices.keys())[0]
    all_indices = list(task_to_indices[task_name])

    # 1. Prepare all samples (shuffle if enabled)
    all_samples = list(all_indices)
    if shuffle_enabled:
        rng = rnd.Random(seed)
        rng.shuffle(all_samples)
        print(f"[Repair Schedule] Shuffled {len(all_samples)} samples before grouping")
    else:
        print(f"[Repair Schedule] {len(all_samples)} samples (no shuffle before grouping)")

    # 2. Group samples: repair_m per group
    repair_groups: List[List[SampleIndex]] = []
    for i in range(0, len(all_samples), repair_m):
        group = all_samples[i:i + repair_m]
        repair_groups.append(group)

    print(f"[Repair Schedule] Created {len(repair_groups)} repair groups (m={repair_m})")

    # 3. For each group, randomly select repair_n samples for reward reversal
    repair_rng = rnd.Random(repair_seed)
    repair_info: Dict[int, Dict[str, Any]] = {}

    schedule: Schedule = []

    for repair_id, group_samples in enumerate(repair_groups, start=1):
        # Randomly select n samples for reversal from this group
        n_to_reverse = min(repair_n, len(group_samples))
        reversed_samples = repair_rng.sample(group_samples, n_to_reverse)

        # Record this repair group's info
        repair_info[repair_id] = {
            "all_samples": group_samples.copy(),
            "reversed_samples": reversed_samples.copy()
        }

        # Add repair group marker to schedule (for identifying group boundaries in main)
        schedule.append((REPAIR_GROUP_MARKER, repair_id))

        print(f"[Repair Schedule] Repair {repair_id}: {len(group_samples)} samples total, {n_to_reverse} reversed")

    print(f"[Repair Schedule] Total repair groups: {len(repair_groups)}")
    return schedule, repair_info


def build_repair_schedule_for_locomo(
    task_name: TaskName,
    locomo_task_instance: Any,
    repair_size_locomo: float,
    repair_seed: int,
    shuffle_enabled: bool,
    seed: int | None,
) -> Tuple[Schedule, Dict[int, Dict[str, Any]]]:
    """
    Build a repair-mode schedule for locomo tasks: partition by session,
    tests the memory system's ability to handle knowledge conflicts.

    For locomo tasks, repair mode partitions by session (repair_m is ignored):
    - Repair 1 = Session 1: randomly select repair_size_locomo * session_qa_count QAs for reversal
    - Repair 2 = Session 2: randomly select repair_size_locomo * session_qa_count QAs for reversal
    - ...

    Each session (repair group) executes 4 phases (same as system memory tasks):
    - wrongJudgeFull: inject session + learn all QAs (with reversed rewards)
    - wrongJudgeStandard: learn only reversed QAs (with reversed rewards)
    - wrongJudgeTestFull: test all QAs (with correct rewards)
    - wrongJudgeTestStandard: test reversed QAs (with correct rewards)
    - rightJudgeFull: re-learn all QAs (with correct rewards)
    - rightJudgeStandard: learn only reversed QAs (with correct rewards)
    - rightJudgeTestFull: test all QAs (with correct rewards)
    - rightJudgeTestStandard: test reversed QAs (with correct rewards)

    Args:
        task_name: locomo task name
        locomo_task_instance: locomo task instance
        repair_size_locomo: fraction of QAs per session to reverse (0-1, e.g. 0.5 = 50%)
        repair_seed: random seed for selecting reversed QAs
        shuffle_enabled: whether to shuffle QAs within each session
        seed: random seed for shuffling

    Returns:
        (schedule, repair_info):
        - schedule: session injection markers + repair group markers
        - repair_info: {repair_id: {"session_id": ..., "all_qa": [...], "reversed_qa": [...]}}
    """
    import random as rnd

    rng = rnd.Random(seed) if shuffle_enabled else None
    repair_rng = rnd.Random(repair_seed)

    schedule: Schedule = []
    repair_info: Dict[int, Dict[str, Any]] = {}

    session_ids = locomo_task_instance.session_ids
    print(f"[Locomo Repair Schedule] Processing {len(session_ids)} sessions: {session_ids}")

    repair_id = 1
    for session_id in session_ids:
        # 1. Get all QA indices for this session
        session_qa_indices = locomo_task_instance.get_qa_indices_for_session(session_id)

        # 2. If shuffle=True, shuffle QA order within this session
        if shuffle_enabled and rng:
            qa_list = list(session_qa_indices)
            rng.shuffle(qa_list)
            session_qa_indices = qa_list

        # 3. Select QAs to reverse based on repair_size_locomo ratio
        n_to_reverse = max(1, int(len(session_qa_indices) * repair_size_locomo))  # Reverse at least 1
        reversed_qa = repair_rng.sample(session_qa_indices, n_to_reverse)

        # 4. Record this repair group (session)'s info
        repair_info[repair_id] = {
            "session_id": session_id,
            "all_qa": list(session_qa_indices),
            "reversed_qa": reversed_qa.copy()
        }

        # 5. Add session injection marker (needed for wrongJudgeFull phase)
        schedule.append((SESSION_INJECTION_MARKER, session_id))

        # 6. Add repair group marker
        schedule.append((REPAIR_GROUP_MARKER, repair_id))

        print(f"[Locomo Repair Schedule] Repair {repair_id} (Session {session_id}): {len(session_qa_indices)} QAs total, {n_to_reverse} reversed ({repair_size_locomo*100:.0f}%)")
        repair_id += 1

    print(f"[Locomo Repair Schedule] Total repair groups: {len(session_ids)}")
    return schedule, repair_info

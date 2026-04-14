"""
Builders module - constructs memory mechanisms, execution engines, and other components.
"""
from __future__ import annotations

from pathlib import Path

from execution.single_agent.single_agent import load_single_agent_engine_from_yaml
from src.runner.config import ExperimentConfig


ROOT_DIR = Path(__file__).resolve().parents[2]


def build_memory_from_config(cfg: ExperimentConfig):
    """
    Construct the memory mechanism from the memory_mechanism section of default.yaml.

    Supported memory mechanisms (all using snake_case names):
    - zero_shot: no-memory baseline
    - stream_icl: streaming In-Context Learning
    - mem0: Mem0 memory system
    - mems: Multi-Memory System (MEMs)
    - awm_pro: Agent Workflow Memory Pro

    Note: regardless of the configured memory_mechanism, the returned instance is used
    for update_memory. In offline mode, use_memory is forced to zero_shot (see main).
    """
    from memory.registry import get_memory_loader, list_available_memories

    mem_cfg = cfg.memory_mechanism or {}
    name = mem_cfg.get("name", "zero_shot")
    config_path = mem_cfg.get("config_path")

    try:
        loader_func, default_config_path = get_memory_loader(name)
    except ValueError as e:
        # Provide a friendly error message
        available = list_available_memories()
        raise ValueError(
            f"Unknown memory mechanism '{name}'. "
            f"Available options: {', '.join(available)}"
        ) from e

    # Determine the config file path
    if not config_path:
        config_path = ROOT_DIR / default_config_path
    else:
        config_path = ROOT_DIR / config_path

    return loader_func(str(config_path))


def build_execution_engine_from_config(cfg: ExperimentConfig):
    """
    Construct the execution engine from the execution_method section of assignment.yaml.
    Supports single_agent.
    """
    exec_cfg = cfg.execution_method or {}
    name = exec_cfg.get("name", "single_agent")
    config_path = exec_cfg.get("config_path")

    if name == "single_agent":
        if not config_path:
            config_path = ROOT_DIR / "execution" / "single_agent" / "single_agent.yaml"
        else:
            config_path = ROOT_DIR / config_path
        return load_single_agent_engine_from_yaml(str(config_path))
    else:
        raise NotImplementedError(f"Execution method '{name}' not implemented yet (supported: single_agent).")


def ensure_output_dir(base: Path) -> Path:
    """Ensure the output directory exists."""
    base.mkdir(parents=True, exist_ok=True)
    return base


def build_schedule_from_config(
    exp_cfg: ExperimentConfig,
    backend,
    locomo_task_instance=None,
    locomo_task_name=None
):
    """
    Unified schedule-building entry point. Returns complete scheduling information.

    Implements a clear pipeline:
    1. Build task indices (build_indices)
    2. Build base schedule (build_base_schedule)
    3. If offline, split into train/test (split_train_test_if_needed)

    Args:
        exp_cfg: experiment configuration
        backend: backend client
        locomo_task_instance: locomo task instance (optional)
        locomo_task_name: locomo task name (optional)

    Returns:
        {
            "train_schedule": [...],      # training schedule
            "test_schedule": [...],       # test schedule (offline mode) or None
            "task_to_indices": {...},     # task index mapping
            "replay_info": {...},         # replay info (replay mode) or None
        }
    """
    # Step 1: Build task indices
    task_to_indices = _build_task_indices(exp_cfg, backend, locomo_task_instance, locomo_task_name)

    # Step 2: Build base schedule
    base_schedule, replay_info = _build_base_schedule(
        exp_cfg=exp_cfg,
        task_to_indices=task_to_indices,
        locomo_task_instance=locomo_task_instance,
        locomo_task_name=locomo_task_name,
    )

    # Step 3: If offline, split into train/test
    training_mode = exp_cfg.experiment.get("training_mode", "offline")
    if training_mode == "offline":
        train_schedule, test_schedule = _split_train_test(
            base_schedule, exp_cfg, locomo_task_instance
        )
    else:
        train_schedule = base_schedule
        test_schedule = None

    return {
        "train_schedule": train_schedule,
        "test_schedule": test_schedule,
        "task_to_indices": task_to_indices,
        "replay_info": replay_info,
    }


def _build_task_indices(exp_cfg: ExperimentConfig, backend, locomo_task_instance=None, locomo_task_name=None) -> dict:
    """
    Step 1: Build the task-to-indices mapping.

    Args:
        exp_cfg: experiment configuration
        backend: backend client
        locomo_task_instance: locomo task instance (optional)
        locomo_task_name: locomo task name (optional)

    Returns:
        {task_name: [sample_indices]}
    """
    from src.runner.schedule_utils import is_locomo_task

    tasks_cfg = exp_cfg.tasks
    task_names = [t["name"] for t in tasks_cfg if "name" in t]

    task_to_indices = {}
    for task_name in task_names:
        # For locomo tasks, get QA indices from the task instance
        if is_locomo_task(task_name) and locomo_task_instance is not None and task_name == locomo_task_name:
            # Locomo task: indices are indices into the QA list
            indices = list(range(len(locomo_task_instance.qa_list)))
            task_to_indices[task_name] = indices
            print(f"[Locomo Task] {task_name}: {len(indices)} QA samples")
        else:
            # Regular task: get indices from backend
            try:
                indices = backend.get_indices(task_name)
                task_to_indices[task_name] = indices
            except Exception as e:
                print(f"Warning: Failed to get indices for task {task_name}: {e}")
                task_to_indices[task_name] = []

    return task_to_indices


def _build_base_schedule(
    exp_cfg: ExperimentConfig,
    task_to_indices: dict,
    locomo_task_instance,
    locomo_task_name: str | None,
):
    """
    Step 2: Build the base schedule based on training mode.

    Args:
        exp_cfg: experiment configuration
        task_to_indices: task index mapping
        locomo_task_instance: locomo task instance
        locomo_task_name: locomo task name

    Returns:
        (schedule, replay_info) tuple
    """
    from src.runner.schedule_utils import (
        build_transfer_schedule,
        build_replay_schedule,
        build_replay_schedule_for_locomo,
        build_repair_schedule,
        build_repair_schedule_for_locomo,
        build_mixed_schedule,
        build_locomo_session_schedule,
        build_offline_locomo_schedule,
    )
    from src.client.scheduler import build_schedule, ScheduleConfig

    # Read config values
    training_mode = exp_cfg.experiment.get("training_mode", "offline")
    cross_task = exp_cfg.experiment.get("cross_task", False)
    shuffle_cfg = exp_cfg.experiment.get("shuffle", {})
    shuffle_enabled = shuffle_cfg.get("enabled", False) if isinstance(shuffle_cfg, dict) else shuffle_cfg
    seed = shuffle_cfg.get("seed") if isinstance(shuffle_cfg, dict) else None

    replay_info = None

    # Build schedule based on training_mode
    if training_mode == "transfer":
        # Transfer mode: train on transfer_task first, then test on transfer_after_task
        transfer_task = exp_cfg.experiment.get("transfer_task")
        transfer_after_task = exp_cfg.experiment.get("transfer_after_task")

        # Check if this is a forward-transfer for a locomo task
        if transfer_task == transfer_after_task and locomo_task_name and locomo_task_instance:
            # Forward transfer + locomo task: use session schedule
            schedule = build_locomo_session_schedule(
                locomo_task_name=locomo_task_name,
                locomo_task_instance=locomo_task_instance,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )
        else:
            # Cross-task transfer or non-locomo task: use default transfer schedule
            schedule = build_transfer_schedule(
                task_to_indices=task_to_indices,
                transfer_task=transfer_task,
                transfer_after_task=transfer_after_task,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )

    elif training_mode == "replay":
        # Replay mode: periodically test already-learned knowledge
        if locomo_task_name:
            # Locomo task replay: partition by session
            schedule, replay_info = build_replay_schedule_for_locomo(
                task_name=locomo_task_name,
                locomo_task_instance=locomo_task_instance,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )
        else:
            # Regular task replay: partition by m/n parameters
            replay_m = exp_cfg.experiment.get("replay_m")
            replay_n = exp_cfg.experiment.get("replay_n")
            replay_seed = exp_cfg.experiment.get("replay_seed")
            schedule, replay_info = build_replay_schedule(
                task_to_indices=task_to_indices,
                replay_m=replay_m,
                replay_n=replay_n,
                replay_seed=replay_seed,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )

    elif training_mode == "repair":
        # Repair mode: test memory system's ability to handle knowledge conflicts
        if locomo_task_name:
            # Locomo task repair: partition by session (uses repair_size_locomo)
            repair_size_locomo = exp_cfg.experiment.get("repair_size_locomo")
            repair_seed = exp_cfg.experiment.get("repair_seed")
            schedule, replay_info = build_repair_schedule_for_locomo(
                task_name=locomo_task_name,
                locomo_task_instance=locomo_task_instance,
                repair_size_locomo=repair_size_locomo,
                repair_seed=repair_seed,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )
        else:
            # Regular task repair: partition by m/n parameters
            repair_m = exp_cfg.experiment.get("repair_m")
            repair_n = exp_cfg.experiment.get("repair_n")
            repair_seed = exp_cfg.experiment.get("repair_seed")
            schedule, replay_info = build_repair_schedule(
                task_to_indices=task_to_indices,
                repair_m=repair_m,
                repair_n=repair_n,
                repair_seed=repair_seed,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )

    elif training_mode == "offline" and locomo_task_name and locomo_task_instance:
        # Offline mode + locomo task: inject all sessions at once, then process all QAs
        schedule = build_offline_locomo_schedule(
            locomo_task_name=locomo_task_name,
            locomo_task_instance=locomo_task_instance,
            shuffle_enabled=shuffle_enabled,
            seed=seed,
        )

    elif training_mode == "online" and locomo_task_name and locomo_task_instance:
        # Online mode + locomo task
        system_memory_tasks = {k: v for k, v in task_to_indices.items() if k != locomo_task_name}
        if system_memory_tasks:
            # Has other tasks: use mixed schedule
            schedule = build_mixed_schedule(
                system_memory_tasks=system_memory_tasks,
                locomo_task_name=locomo_task_name,
                locomo_task_instance=locomo_task_instance,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )
        else:
            # Only locomo task: use session schedule
            schedule = build_locomo_session_schedule(
                locomo_task_name=locomo_task_name,
                locomo_task_instance=locomo_task_instance,
                shuffle_enabled=shuffle_enabled,
                seed=seed,
            )

    else:
        # Default: use scheduler's build_schedule (for regular system memory tasks)
        schedule_cfg = ScheduleConfig(
            cross_task=cross_task,
            shuffle=shuffle_enabled,
            seed=seed,
        )
        schedule = build_schedule(task_to_indices, schedule_cfg)

    return schedule, replay_info


def _split_train_test(schedule, exp_cfg: ExperimentConfig, locomo_task_instance=None):
    """
    Step 3: If in offline mode, split the schedule into train/test.

    For locomo tasks:
    - train set = all session injection markers (for injecting context into memory)
    - test set = all QAs (for testing memory retrieval)
    - train_size parameter is ignored

    For regular tasks:
    - split by train_size ratio

    Args:
        schedule: base schedule sequence
        exp_cfg: experiment configuration
        locomo_task_instance: locomo task instance (optional)

    Returns:
        (train_schedule, test_schedule) tuple
    """
    from src.runner.schedule_utils import SESSION_INJECTION_MARKER

    # Check if this is a locomo offline scenario
    if locomo_task_instance is not None:
        # Locomo mode: train = sessions, test = all QAs
        # Split point = number of sessions
        split_point = len(locomo_task_instance.session_ids)
        train_schedule = schedule[:split_point]
        test_schedule = schedule[split_point:]

        print(f"[Offline Locomo Mode] Split schedule:")
        print(f"  - Train: {len(train_schedule)} session injections")
        print(f"  - Test: {len(test_schedule)} QAs")
        print(f"  - train_size parameter is IGNORED for locomo tasks")
    else:
        # Regular task: split by train_size ratio
        train_size = exp_cfg.experiment.get("train_size", 0.8)
        split_point = int(len(schedule) * train_size)
        train_schedule = schedule[:split_point]
        test_schedule = schedule[split_point:]

        print(f"[Offline Mode] Split schedule: train={len(train_schedule)}, test={len(test_schedule)} (train_size={train_size})")

    return train_schedule, test_schedule

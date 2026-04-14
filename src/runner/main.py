from __future__ import annotations

"""
End-to-end runner for the lifelong-learning benchmark (many memory method + single_agent, multi-turn, tool-calling).

当前版本：
- 串起 assignment.yaml / scheduler / backend / memory.zero_shot / execution.single_agent
- 真实调用 LLM（OpenAI 风格接口），传递 tools，支持 tool_calls，多轮 /interact 直到后端结束

运行方式（在项目根目录）：
    python -m src.runner.main
"""

import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from execution.single_agent.single_agent import SingleAgentExecutionEngine
from memory.zero_shot.zero_shot import load_zero_shot_from_yaml
from src.client.scheduler import ScheduleConfig, build_schedule, TaskName, SampleIndex, Schedule
from src.runner.agent import SimpleHTTPChatAgent
from src.runner.backend import BackendClient
from src.runner.builders import build_memory_from_config, build_execution_engine_from_config, ensure_output_dir, build_schedule_from_config
from src.runner.config import ExperimentConfig, load_experiment_config, ROOT_DIR
from src.runner.schedule_utils import (
    load_task_instance, is_locomo_task,
    SESSION_INJECTION_MARKER, REPLAY_TEST_MARKER, REPAIR_GROUP_MARKER
)
from src.server.tasks.locomo.task import convert_session_to_history

# 默认后端地址，可通过环境变量覆盖
BACKEND_BASE_URL = os.getenv("LLBENCH_BACKEND_URL", "http://localhost:5038/api")


class LocomoSessionWrapper:
    """
    Locomo 任务的 Session 包装器，直接调用 LLM agent

    用于在不需要完整 Session 交互的情况下，简化 locomo 任务的执行流程
    """
    def __init__(self, session_id: int, llm_agent, memory_for_enhance, task_name, locomo_task_instance, training_mode: str = "online"):
        from src.server.tasks.locomo.task_base import Session

        # 继承 Session 的初始化
        self.session_id = session_id
        self.id = session_id  # locomo task 代码需要 session.id
        self.history = []
        self.llm_agent = llm_agent
        self.memory_for_enhance = memory_for_enhance
        self.task_name = task_name
        self.locomo_task_instance = locomo_task_instance
        self.training_mode = training_mode
        self._loop = None
        self._empty_response_retry_limit = 3

    def inject(self, messages):
        """注入消息到 history"""
        if isinstance(messages, list):
            self.history.extend(messages)
        else:
            self.history.append(messages)

    def sync_action(self, *injection):
        """直接调用 LLM agent，不需要复杂的 Session 交互"""
        from src.server.tasks.locomo.task_base import AgentOutput, AgentOutputStatus
        from openai.types.chat import (
            ChatCompletionSystemMessageParam,
            ChatCompletionUserMessageParam,
            ChatCompletionAssistantMessageParam
        )

        # 注入消息
        self.inject(list(injection))

        # 将 history 转换为 messages 格式（只包含 system, user, assistant）
        messages = []
        for item in self.history:
            if hasattr(item, 'root'):
                msg = item.root
            elif isinstance(item, dict):
                msg = item
            else:
                continue

            # 只包含聊天消息，排除 RewardHistoryItem
            if msg.get("role") in ["system", "user", "assistant"]:
                messages.append(msg)

        # 对于 zero-shot + locomo 任务，需要特殊处理：
        # 将当前 QA 所属的 session 以 user 角色插入到 system prompt 和当前问题之间
        if self.memory_for_enhance is not None:
            from memory.zero_shot.zero_shot import ZeroShotMemory

            # 检查是否是 zero-shot 方法
            is_zero_shot = isinstance(self.memory_for_enhance, ZeroShotMemory)

            if is_zero_shot and is_locomo_task(self.task_name) and self.locomo_task_instance is not None:
                # Zero-shot + locomo：根据 where_ground_truth 插入对应的 session(s)
                # where_ground_truth 是一个 list，包含该问题需要参考的所有 session id
                sample_index = self.session_id  # session_id 在这里实际上是 sample_index
                if sample_index < len(self.locomo_task_instance.qa_list):
                    qa_item = self.locomo_task_instance.qa_list[sample_index]
                    where_ground_truth = qa_item.get("where_ground_truth", [])

                    # 如果没有 where_ground_truth，回退到使用 where（单个 session）
                    if not where_ground_truth:
                        current_session_id = qa_item.get("where")
                        if current_session_id is not None:
                            where_ground_truth = [current_session_id]

                    if where_ground_truth:
                        # Determine which sessions to inject based on training_mode:
                        # - offline: inject ALL sessions (model has seen all conversations before testing)
                        # - online/transfer/replay/repair: inject sessions 1..max(where_ground_truth),
                        #   simulating what the agent has accumulated in the conversation stream so far
                        all_session_ids = self.locomo_task_instance.session_ids
                        if self.training_mode == "offline":
                            sessions_to_inject = all_session_ids
                        else:
                            max_session = max(where_ground_truth)
                            sessions_to_inject = [s for s in all_session_ids if s <= max_session]

                        session_messages = []
                        for session_id in sessions_to_inject:
                            session_history = self.locomo_task_instance.get_session_history(session_id)
                            for hist_item in session_history:
                                session_messages.append({
                                    "role": "user",
                                    "content": hist_item.get("content", "")
                                })

                        # 插入位置：system prompt 之后，当前问题之前
                        # messages 结构：[system_prompt, current_question]
                        if len(messages) >= 2 and messages[0].get("role") == "system":
                            # 在 system prompt 和 question 之间插入 session(s)
                            messages = [messages[0]] + session_messages + messages[1:]
                            print(f"  -> [Zero-shot + Locomo] Injected {len(sessions_to_inject)} session(s) {sessions_to_inject} ({len(session_messages)} messages) for QA {sample_index} (mode={self.training_mode})")

                            # 同时将这些历史session消息注入到self.history中，以便保存时包含它们
                            # 插入位置：在system prompt之后，current question之前
                            # 找到self.history中system prompt的位置
                            system_idx = -1
                            for i, item in enumerate(self.history):
                                if hasattr(item, 'root'):
                                    msg = item.root
                                elif isinstance(item, dict):
                                    msg = item
                                else:
                                    continue

                                if msg.get("role") == "system":
                                    system_idx = i
                                    break

                            # 在system prompt之后插入历史session消息
                            if system_idx >= 0:
                                insert_position = system_idx + 1
                                for session_msg in session_messages:
                                    self.history.insert(insert_position, ChatCompletionUserMessageParam(
                                        role="user",
                                        content=session_msg.get("content", "")
                                    ))
                                    insert_position += 1

            # single_agent
            enhanced_messages = self.memory_for_enhance.use_memory(self.task_name, messages)

            # 将增强后的消息更新到 history 中（以便保存时包含增强内容）
            # 检测 messages 和 enhanced_messages 的差异
            if enhanced_messages != messages:
                # 遍历 enhanced_messages，找出被修改的消息并更新到 history
                for idx, (orig_msg, enhanced_msg) in enumerate(zip(messages, enhanced_messages)):
                    # 检查 content 是否被修改
                    if orig_msg.get("content") != enhanced_msg.get("content"):
                        # 找到 history 中对应的消息并更新
                        history_idx = 0
                        msg_count = 0
                        for i, item in enumerate(self.history):
                            if hasattr(item, 'root'):
                                msg = item.root
                            elif isinstance(item, dict):
                                msg = item
                            else:
                                continue

                            # 只计数 system/user/assistant 消息
                            if msg.get("role") in ["system", "user", "assistant"]:
                                if msg_count == idx:
                                    history_idx = i
                                    break
                                msg_count += 1

                        # 更新 history 中的消息
                        if history_idx < len(self.history):
                            role = enhanced_msg.get("role")
                            content = enhanced_msg.get("content", "")
                            if role == "system":
                                self.history[history_idx] = ChatCompletionSystemMessageParam(
                                    role="system",
                                    content=content
                                )
                            elif role == "user":
                                self.history[history_idx] = ChatCompletionUserMessageParam(
                                    role="user",
                                    content=content
                                )
        else:
            enhanced_messages = messages

        # 直接调用 LLM agent
        agent = self.llm_agent
        assistant_messages = []
        for attempt in range(self._empty_response_retry_limit):
            response = agent.inference(enhanced_messages, tools=None)
            content = response.get("content")
            if content:
                assistant_msg = {
                    "role": "assistant",
                    "content": content
                }
                assistant_messages.append(assistant_msg)
                self.inject(ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=content
                ))
                break

            logging.warning(
                "[LocomoSessionWrapper] Empty assistant response for session %s "
                "(attempt %s/%s); retrying...",
                self.session_id,
                attempt + 1,
                self._empty_response_retry_limit,
            )
            time.sleep(min(2 * (attempt + 1), 5))

        return AgentOutput(
            status=AgentOutputStatus.NORMAL,
            messages=assistant_messages
        )


def validate_training_mode_constraints(exp_cfg: ExperimentConfig) -> tuple[str, bool]:
    """
    集中校验训练模式的约束条件

    Args:
        exp_cfg: 实验配置

    Returns:
        (training_mode, cross_task): 训练模式和跨任务标志

    Raises:
        ValueError: 当配置不满足训练模式的约束时
    """
    training_mode = exp_cfg.experiment.get("training_mode", "offline")
    cross_task = exp_cfg.experiment.get("cross_task", False)
    tasks_cfg = exp_cfg.tasks
    task_names: List[str] = [t["name"] for t in tasks_cfg if "name" in t]

    # 检查是否有多个 locomo 任务（personal memory 数据集只能有一个）
    locomo_tasks = [name for name in task_names if is_locomo_task(name)]
    if len(locomo_tasks) > 1:
        raise ValueError(
            f"Multiple personal memory tasks (locomo) detected: {locomo_tasks}. "
            "Only one personal memory task (locomo-0 - locomo-9) is allowed per run."
        )

    if training_mode == "transfer":
        # transfer 模式：分为两种情况
        transfer_task = exp_cfg.experiment.get("transfer_task")
        transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
        if not transfer_task or not transfer_after_task:
            raise ValueError("transfer mode requires both transfer_task and transfer_after_task to be set")

        if transfer_task != transfer_after_task:
            # 情况1：跨任务迁移（transfer_task != transfer_after_task）
            # 必须 cross_task=True，必须选中两个任务，不允许 locomo 任务
            if not cross_task:
                raise ValueError("transfer mode with different tasks requires cross_task=True")
            if len(task_names) != 2:
                raise ValueError(
                    f"transfer mode with different tasks requires exactly 2 tasks, but found {len(task_names)} tasks: {task_names}"
                )
            if locomo_tasks:
                raise ValueError(
                    f"transfer mode with different tasks does not support personal memory tasks (locomo). "
                    f"Found locomo task(s): {locomo_tasks}"
                )
            if transfer_task not in task_names or transfer_after_task not in task_names:
                raise ValueError(
                    f"transfer mode: transfer_task={transfer_task} and transfer_after_task={transfer_after_task} "
                    f"must be in the selected tasks: {task_names}"
                )
        else:
            # 情况2：前向迁移（transfer_task == transfer_after_task）
            # 可以是任意任务（包括locomo），必须只选中一个任务，必须设置 forward_transfer_num
            if len(task_names) != 1:
                raise ValueError(
                    f"transfer mode with same task requires exactly 1 task, but found {len(task_names)} tasks: {task_names}"
                )
            if transfer_task not in task_names:
                raise ValueError(
                    f"transfer mode: transfer_task={transfer_task} must be in the selected tasks: {task_names}"
                )
            forward_transfer_num = exp_cfg.experiment.get("forward_transfer_num")
            if forward_transfer_num is None or forward_transfer_num <= 0:
                raise ValueError(
                    f"transfer mode with same task requires forward_transfer_num to be set and > 0. "
                    f"Got: forward_transfer_num={forward_transfer_num}"
                )
    elif training_mode == "replay":
        # replay 模式：必须 cross_task=False，必须只选中一个任务
        if cross_task:
            raise ValueError("replay mode requires cross_task=False")
        if len(task_names) != 1:
            raise ValueError(
                f"replay mode requires exactly 1 task, but found {len(task_names)} tasks: {task_names}"
            )
        # 检查 replay 参数是否设置（对于非 locomo 任务）
        if not locomo_tasks:
            replay_m = exp_cfg.experiment.get("replay_m")
            replay_n = exp_cfg.experiment.get("replay_n")
            replay_seed = exp_cfg.experiment.get("replay_seed")
            if replay_m is None or replay_n is None or replay_seed is None:
                raise ValueError(
                    f"replay mode requires replay_m, replay_n, and replay_seed to be set. "
                    f"Got: replay_m={replay_m}, replay_n={replay_n}, replay_seed={replay_seed}"
                )
    elif training_mode == "repair":
        # repair 模式：必须 cross_task=False，必须只选中一个任务
        if cross_task:
            raise ValueError("repair mode requires cross_task=False")
        if len(task_names) != 1:
            raise ValueError(
                f"repair mode requires exactly 1 task, but found {len(task_names)} tasks: {task_names}"
            )
        # 检查 repair 参数是否设置
        if locomo_tasks:
            # locomo 任务：需要 repair_size_locomo 和 repair_seed
            repair_size_locomo = exp_cfg.experiment.get("repair_size_locomo")
            repair_seed = exp_cfg.experiment.get("repair_seed")
            if repair_size_locomo is None or repair_seed is None:
                raise ValueError(
                    f"repair mode for locomo tasks requires repair_size_locomo and repair_seed to be set. "
                    f"Got: repair_size_locomo={repair_size_locomo}, repair_seed={repair_seed}"
                )
            if not (0 < repair_size_locomo <= 1):
                raise ValueError(
                    f"repair_size_locomo must be between 0 and 1 (exclusive 0, inclusive 1). "
                    f"Got: repair_size_locomo={repair_size_locomo}"
                )
        else:
            # 非 locomo 任务：需要 repair_m, repair_n, repair_seed
            repair_m = exp_cfg.experiment.get("repair_m")
            repair_n = exp_cfg.experiment.get("repair_n")
            repair_seed = exp_cfg.experiment.get("repair_seed")
            if repair_m is None or repair_n is None or repair_seed is None:
                raise ValueError(
                    f"repair mode requires repair_m, repair_n, and repair_seed to be set. "
                    f"Got: repair_m={repair_m}, repair_n={repair_n}, repair_seed={repair_seed}"
                )
    elif training_mode == "offline":
        # offline 模式：必须 cross_task=False，必须只选中一个任务
        if cross_task:
            raise ValueError("offline mode requires cross_task=False")
        if len(task_names) != 1:
            raise ValueError(
                f"offline mode requires exactly 1 task, but found {len(task_names)} tasks: {task_names}"
            )
    else:
        # online 模式：验证 cross_task 和任务数量的一致性
        if not cross_task:
            # cross_task=False 时必须只能选中一个数据集
            if len(task_names) != 1:
                raise ValueError(
                    f"Invalid configuration: cross_task=False requires exactly 1 task, "
                    f"but found {len(task_names)} tasks: {task_names}"
                )
        else:
            # cross_task=True 时必须选中大于一个数据集
            if len(task_names) <= 1:
                raise ValueError(
                    f"Invalid configuration: cross_task=True requires more than 1 task, "
                    f"but found {len(task_names)} task(s): {task_names}"
                )

    return training_mode, cross_task


def main() -> None:
    print(f"Using backend base URL: {BACKEND_BASE_URL}")
    backend = BackendClient(BACKEND_BASE_URL)

    # 1) 简单健康检查
    try:
        workers = backend.list_workers()
        print("Controller /list_workers OK. Available tasks:")
        print(json.dumps(workers, indent=2))
    except Exception as e:
        print(f"Failed to call /list_workers: {e}")
        print("请确认后端 Controller 已在默认端口 5038 启动，或通过 LLBENCH_BACKEND_URL 覆盖地址。")
        return

    # 2) 读取 assignment 配置
    exp_cfg = load_experiment_config()

    # 2.1) 校验训练模式约束（集中校验），并获取 training_mode 和 cross_task
    training_mode, cross_task = validate_training_mode_constraints(exp_cfg)

    # 2.2) 获取 shuffle 配置
    shuffle_cfg = exp_cfg.experiment.get("shuffle", {})
    shuffle_enabled = shuffle_cfg.get("enabled", False) if isinstance(shuffle_cfg, dict) else shuffle_cfg

    # 3) 检测并加载 locomo 任务（需要在构建调度之前）
    locomo_task_instance = None
    locomo_task_name = None
    tasks_cfg = exp_cfg.tasks
    task_names: List[str] = [t["name"] for t in tasks_cfg if "name" in t]

    # 检查是否有 locomo 任务
    locomo_tasks = [name for name in task_names if is_locomo_task(name)]

    # 如果有 locomo 任务，加载它
    if len(locomo_tasks) == 1:
        task_name = locomo_tasks[0]
        locomo_task_instance = load_task_instance(task_name, exp_cfg)
        locomo_task_name = task_name
        if locomo_task_instance is None:
            raise ValueError(f"Failed to load locomo task instance for {task_name}")
        print(f"\n[Locomo Task Detected] {task_name}, sessions: {locomo_task_instance.session_ids}")

    # 4) 构造调度序列（统一入口，返回完整的调度信息）
    schedule_result = build_schedule_from_config(
        exp_cfg, backend,
        locomo_task_instance=locomo_task_instance,
        locomo_task_name=locomo_task_name
    )

    train_schedule = schedule_result["train_schedule"]
    test_schedule = schedule_result["test_schedule"]
    task_to_indices = schedule_result["task_to_indices"]
    replay_info = schedule_result["replay_info"]

    print("\nTasks and available indices:")
    for task, indices in task_to_indices.items():
        print(f"  {task}: {len(indices)} indices -> {indices[:10]}{' ...' if len(indices) > 10 else ''}")

    print(f"\nSchedule summary:")
    print(f"  Train schedule: {len(train_schedule)} samples")
    if test_schedule:
        print(f"  Test schedule: {len(test_schedule)} samples")
    print(f"  First 20 train entries:")
    for pair in train_schedule[:20]:
        print("   ", pair)

    if not train_schedule:
        print("Train schedule is empty; nothing to run.")
        return

    # 4) 构造 memory + execution engine
    execution_engine = build_execution_engine_from_config(exp_cfg)

    def build_memory_bundle():
        """按执行方式与训练模式构建 memory 与 memory_for_enhance，便于任务切换时重置。"""
        mem = build_memory_from_config(exp_cfg)
        mem_enh = None

        if training_mode == "offline":
            mem_enh = load_zero_shot_from_yaml(str(ROOT_DIR / "memory" / "zero_shot" / "zero_shot.yaml"))
            print(f"Training mode: {training_mode} -> Using zero_shot for use_memory (memory disabled), but still updating memory with {exp_cfg.memory_mechanism.get('name', 'zero_shot')}")
        elif training_mode in ("online", "transfer", "replay", "repair"):
            # online, transfer, replay, repair 模式都使用配置的记忆机制
            mem_enh = mem
            print(f"Training mode: {training_mode} -> Using {exp_cfg.memory_mechanism.get('name', 'zero_shot')} for both use_memory and update_memory")
        else:
            raise ValueError(f"Unknown training_mode: {training_mode} (must be 'online', 'offline', 'transfer', 'replay', or 'repair')")

        return mem, mem_enh

    # 初始 memory
    memory, memory_for_enhance = build_memory_bundle()

    # 4.1) locomo 任务的 session 注入统一由 schedule 中的 SESSION_INJECTION_MARKER 驱动
    # 不再在 offline 模式下预注入，避免重复注入
    # Online/Offline 模式的 session 注入都会在训练循环中通过 marker 触发

    # 5) 输出目录（根据 train_size 分割，创建 train/test 子目录）
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_output_root = ensure_output_dir(ROOT_DIR / "outputs" / timestamp)

    # 根据 training_mode 确定子目录名称
    if test_schedule:
        if training_mode == "transfer":
            # transfer 模式（跨任务）：使用 transfer_train 和 transfer_test 目录名
            train_output_root = ensure_output_dir(base_output_root / "transfer_train")
            test_output_root = ensure_output_dir(base_output_root / "transfer_test")
        else:
            # offline 模式：使用 train 和 test 目录名
            train_output_root = ensure_output_dir(base_output_root / "train")
            test_output_root = ensure_output_dir(base_output_root / "test")
    else:
        if training_mode == "transfer":
            # transfer 模式（前向迁移）：使用 transfer_train 目录名
            train_output_root = ensure_output_dir(base_output_root / "transfer_train")
        else:
            train_output_root = base_output_root
        test_output_root = None

    # 为 execution engine 构造 LLM agent(s)（基于 llmapi 配置）
    if isinstance(execution_engine, SingleAgentExecutionEngine):
        # single_agent: 创建一个 agent
        llm_agent = SimpleHTTPChatAgent(execution_engine.config.agent_name)
    else:
        llm_agent = None

    # 6) 执行训练集样本
    last_task_name: TaskName | None = None
    
    # 记录执行顺序（用于绘制正确率随时间变化的图像）
    # online模式：在任务目录下保存 execution_order.json
    # offline模式：在train和test目录下分别保存 execution_order.json
    execution_order_train: Dict[TaskName, List[Dict[str, Any]]] = {}  # {task_name: [execution_record, ...]}
    execution_order_test: Dict[TaskName, List[Dict[str, Any]]] = {}   # {task_name: [execution_record, ...]}
    execution_order_forward_test: Dict[TaskName, List[Dict[str, Any]]] = {}  # transfer模式的前向迁移测试

    # Replay 模式：跟踪当前 replay 状态
    current_replay_id = 0
    learned_samples_in_replay: List[SampleIndex] = []  # 当前已学习的样本（用于确定 replay_id）
    current_replay_id_for_test = 1 if (training_mode == "replay" and replay_info) else 0  # 当前正在执行的 replay 的测试阶段（用于确定测试样本应该保存到哪个 replay）

    # =========================================================================
    # Repair 模式：测试记忆系统处理知识冲突的能力
    # =========================================================================
    if training_mode == "repair" and replay_info:
        print(f"\n{'='*80}")
        print(f"Running REPAIR mode: {len(replay_info)} repair groups")
        print(f"{'='*80}\n")

        # 获取任务名称（repair 模式只支持单任务）
        if len(task_to_indices) != 1:
            raise ValueError(f"repair mode requires exactly 1 task, but got {len(task_to_indices)} tasks")
        actual_task_name = list(task_to_indices.keys())[0]

        # 处理每个 repair 组
        for repair_id, repair_group_info in replay_info.items():
            print(f"\n{'='*80}")
            print(f"Processing Repair Group {repair_id}")
            print(f"{'='*80}\n")

            # 获取该 repair 组的信息
            if is_locomo_task(actual_task_name):
                # Locomo 任务：repair_group_info = {"session_id": ..., "all_qa": [...], "reversed_qa": [...]}
                session_id = repair_group_info.get("session_id")
                all_samples = repair_group_info.get("all_qa", [])
                reversed_samples = repair_group_info.get("reversed_qa", [])

                # Locomo 任务：首先注入 session
                if locomo_task_instance and locomo_task_name:
                    print(f"[Repair {repair_id}] Injecting session {session_id} content into memory...")
                    session_history = locomo_task_instance.get_session_history(session_id)
                    if session_history:
                        if isinstance(memory, dict):
                            for agent_name, agent_mem in memory.items():
                                agent_mem.update_memory(locomo_task_name, session_history, {"session_id": session_id, "type": "session_injection", "reward": 1, "status": "completed"})
                        else:
                            memory.update_memory(locomo_task_name, session_history, {"session_id": session_id, "type": "session_injection", "reward": 1, "status": "completed"})
                        print(f"  -> Injected session {session_id} ({len(session_history)} dialogues)")
            else:
                # 普通任务：repair_group_info = {"all_samples": [...], "reversed_samples": [...]}
                all_samples = repair_group_info.get("all_samples", [])
                reversed_samples = repair_group_info.get("reversed_samples", [])

            print(f"[Repair {repair_id}] All samples: {len(all_samples)}, Reversed samples: {len(reversed_samples)}")

            # 创建 repair 组输出目录
            repair_dir = base_output_root / f"repair{repair_id}"
            repair_dir.mkdir(parents=True, exist_ok=True)

            # 定义 4 个阶段
            phases = [
                {"name": "wrongJudge", "use_reversed_rewards": True, "is_test": False},
                {"name": "wrongJudgeTest", "use_reversed_rewards": False, "is_test": True},
                {"name": "rightJudge", "use_reversed_rewards": False, "is_test": False},
                {"name": "rightJudgeTest", "use_reversed_rewards": False, "is_test": True},
            ]

            # 执行 4 个阶段
            for phase in phases:
                phase_name = phase["name"]
                use_reversed_rewards = phase["use_reversed_rewards"]
                is_test_phase = phase["is_test"]

                print(f"\n{'─'*60}")
                print(f"[Repair {repair_id}] Phase: {phase_name}")
                print(f"  - Reversed rewards: {use_reversed_rewards}")
                print(f"  - Test mode: {is_test_phase}")
                print(f"{'─'*60}\n")

                # 创建阶段输出目录
                phase_full_dir = repair_dir / f"{phase_name}Full" / actual_task_name
                phase_standard_dir = repair_dir / f"{phase_name}Standard" / actual_task_name
                phase_full_dir.mkdir(parents=True, exist_ok=True)
                phase_standard_dir.mkdir(parents=True, exist_ok=True)

                # 执行所有样本（保存到 Full）
                for sample_idx in all_samples:
                    is_reversed = sample_idx in reversed_samples

                    try:
                        # 对于 locomo 任务，使用任务实例
                        if is_locomo_task(actual_task_name) and locomo_task_instance and locomo_task_name == actual_task_name:
                            session = LocomoSessionWrapper(sample_idx, llm_agent, memory_for_enhance, actual_task_name, locomo_task_instance, training_mode)
                            task_result = locomo_task_instance.sync_start_sample(sample_idx, session)

                            # 提取 messages 和 reward
                            messages = []
                            for item in session.history:
                                if hasattr(item, 'root') and isinstance(item.root, dict):
                                    msg = item.root
                                    if msg.get("role") in ["system", "user", "assistant"]:
                                        messages.append(msg)
                                elif isinstance(item, dict) and item.get("role") in ["system", "user", "assistant"]:
                                    messages.append(item)

                            reward = 0
                            for item in session.history:
                                if hasattr(item, 'root') and hasattr(item.root, 'reward'):
                                    reward_item = item.root
                                    if hasattr(reward_item, 'metrics') and isinstance(reward_item.metrics, dict):
                                        llm_score = reward_item.metrics.get("llm_score")
                                        if llm_score is not None:
                                            reward = float(llm_score)
                                            break
                                    reward = reward_item.reward
                                    break
                                elif isinstance(item, dict) and "reward" in item:
                                    if "metrics" in item and isinstance(item["metrics"], dict):
                                        llm_score = item["metrics"].get("llm_score")
                                        if llm_score is not None:
                                            reward = float(llm_score)
                                            break
                                    reward = item["reward"]
                                    break

                            # 如果从 history 中没有找到，尝试从 task_result 中获取
                            if reward == 0 and isinstance(task_result.result, dict):
                                metrics = task_result.result.get("metrics")
                                if isinstance(metrics, dict):
                                    llm_score = metrics.get("llm_score")
                                    if llm_score is not None:
                                        reward = float(llm_score)

                            # 调试：打印反转前的值
                            original_reward = reward

                            # 应用奖励反转
                            if use_reversed_rewards and is_reversed:
                                reward = 1 - reward  # 反转奖励：0->1, 1->0
                                print(f"    [DEBUG] Sample {sample_idx}: original_reward={original_reward:.2f}, is_reversed={is_reversed}, use_reversed_rewards={use_reversed_rewards}, final_reward={reward:.2f}")

                            result = {
                                "reward": reward,
                                "status": task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status),
                                "result": task_result.result
                            }

                        else:
                            # 普通任务：使用后端
                            sess = backend.start_sample(actual_task_name, sample_idx)
                            session_id_backend = sess["session_id"]

                            # 获取初始输入
                            obs = backend.get_observation(session_id_backend)
                            messages = obs.get("history", [])

                            # 执行任务
                            while True:
                                enhanced_msg = execution_engine.run(messages, memory_for_enhance)
                                messages.append(enhanced_msg)

                                step_result = backend.step(session_id_backend, enhanced_msg)
                                if step_result.get("done", False):
                                    break

                                obs = backend.get_observation(session_id_backend)
                                obs_msg = obs.get("observation")
                                if obs_msg:
                                    messages.append(obs_msg)

                            result = backend.get_result(session_id_backend)
                            reward = result.get("reward", 0)

                            # 调试：打印反转前的值
                            original_reward = reward

                            # 应用奖励反转
                            if use_reversed_rewards and is_reversed:
                                reward = 1 - reward  # 反转奖励：0->1, 1->0
                                print(f"    [DEBUG] Sample {sample_idx}: original_reward={original_reward:.2f}, is_reversed={is_reversed}, use_reversed_rewards={use_reversed_rewards}, final_reward={reward:.2f}")

                            result["reward"] = reward

                        # 更新 memory（除非是测试阶段）
                        if not is_test_phase:
                            if isinstance(memory, dict):
                                for agent_name, agent_mem in memory.items():
                                    agent_mem.update_memory(actual_task_name, messages, result)
                            else:
                                memory.update_memory(actual_task_name, messages, result)

                        # 保存结果到 Full 目录
                        sample_file_full = phase_full_dir / f"{sample_idx}.json"
                        with open(sample_file_full, 'w', encoding='utf-8') as f:
                            json.dump({"messages": messages, "result": result}, f, ensure_ascii=False, indent=2)

                        # 如果是 reversed 样本，也保存到 Standard 目录
                        if is_reversed:
                            sample_file_standard = phase_standard_dir / f"{sample_idx}.json"
                            with open(sample_file_standard, 'w', encoding='utf-8') as f:
                                json.dump({"messages": messages, "result": result}, f, ensure_ascii=False, indent=2)

                        status_marker = "[REVERSED]" if is_reversed else ""
                        test_marker = "[TEST]" if is_test_phase else ""
                        print(f"  {status_marker}{test_marker} Sample {sample_idx}: reward={reward:.2f}")

                    except Exception as e:
                        print(f"  [ERROR] Sample {sample_idx} failed: {e}")
                        import traceback
                        traceback.print_exc()

                print(f"\n[Repair {repair_id}] {phase_name} completed:")
                print(f"  - Full: {len(all_samples)} samples saved to {phase_full_dir}")
                print(f"  - Standard: {len(reversed_samples)} samples saved to {phase_standard_dir}")

        print(f"\n{'='*80}")
        print(f"Repair mode execution completed")
        print(f"{'='*80}\n")
        return  # Repair 模式执行完毕，直接返回

    # =========================================================================
    # 正常训练/测试执行（非 repair 模式）
    # =========================================================================

    if train_schedule:
        print(f"\n{'='*60}")
        if training_mode == "transfer":
            # Transfer 模式：统计训练样本和测试样本的数量
            transfer_task = exp_cfg.experiment.get("transfer_task")
            transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
            train_count = sum(1 for task_name, _ in train_schedule if task_name == transfer_task)
            test_count = sum(1 for task_name, _ in train_schedule if task_name == transfer_after_task)
            print(f"Running Transfer mode: {len(train_schedule)} samples (train={train_count}, test={test_count})")
            print(f"  TRAIN task: {transfer_task} ({train_count} samples)")
            print(f"  TEST task: {transfer_after_task} ({test_count} samples)")
        else:
            print(f"Running TRAIN set: {len(train_schedule)} samples")
        print(f"{'='*60}\n")
        
        for idx, (task_name, sample_index) in enumerate(train_schedule, start=1):
            # 处理 replay 模式的测试样本标记
            is_replay_test = False
            if task_name == REPLAY_TEST_MARKER:
                # replay 模式的测试样本：需要从 task_to_indices 中获取实际的任务名称
                if len(task_to_indices) != 1:
                    raise ValueError(f"replay mode: expected 1 task, but got {len(task_to_indices)} tasks")
                actual_task_name = list(task_to_indices.keys())[0]
                task_name = actual_task_name
                is_replay_test = True
                # 使用 current_replay_id_for_test 来确定当前是哪个 replay 的测试阶段
                # 这个值在训练样本处理时会被更新
                current_replay_id = current_replay_id_for_test
                print(f"[TRAIN {idx}/{len(train_schedule)}] [REPLAY TEST] task={task_name}, index={sample_index} (replay{current_replay_id})")
            else:
                # 训练样本：确定当前 replay_id（根据训练样本所属的 replay）
                if training_mode == "replay" and replay_info:
                    for rid, info in replay_info.items():
                        if sample_index in info["train"]:
                            # 找到包含该训练样本的最大 replay_id（即最新的 replay）
                            # 因为训练样本会出现在多个 replay 的 train 列表中（累积的）
                            current_replay_id = max(current_replay_id, rid)
                            break
            
            # 处理 session 注入标记（用于混合调度）
            if task_name == SESSION_INJECTION_MARKER:
                session_id = sample_index  # 在混合调度中，sample_index 存储的是 session_id
                if locomo_task_instance is not None and locomo_task_name:
                    print(f"[TRAIN {idx}/{len(train_schedule)}] [SESSION INJECTION] Injecting session {session_id} content into memory...")
                    session_history = locomo_task_instance.get_session_history(session_id)
                    if session_history:
                        if isinstance(memory, dict):
                            # Multi-agent: 更新所有 agent 的 memory
                            for agent_name, agent_mem in memory.items():
                                agent_mem.update_memory(locomo_task_name, session_history, {"session_id": session_id, "type": "session_injection", "reward": 1, "status": "completed"})
                        else:
                            memory.update_memory(locomo_task_name, session_history, {"session_id": session_id, "type": "session_injection", "reward": 1, "status": "completed"})
                        print(f"  -> Injected session {session_id} ({len(session_history)} dialogues)")
                    else:
                        print(f"  -> Warning: Session {session_id} has no history")
                else:
                    print(f"  -> Warning: SESSION_INJECTION_MARKER found but locomo_task_instance is None")
                continue  # 跳过执行，继续下一个样本

            # 如果不跨任务学习且任务切换，重置 memory
            if not cross_task and last_task_name is not None and task_name != last_task_name:
                memory, memory_for_enhance = build_memory_bundle()
                print(f"\n[Memory Reset] cross_task=False, switched task {last_task_name} -> {task_name}, memory rebuilt.\n")
            last_task_name = task_name
            print(f"[TRAIN {idx}/{len(train_schedule)}] task={task_name}, index={sample_index}")

            try:
                # 对于 locomo 任务，直接使用任务实例，不需要后端
                if is_locomo_task(task_name) and locomo_task_instance is not None and locomo_task_name == task_name:
                    # 创建包装的 Session
                    session = LocomoSessionWrapper(sample_index, llm_agent, memory_for_enhance, task_name, locomo_task_instance, training_mode)
                    
                    # 直接调用任务实例的 sync_start_sample
                    task_result = locomo_task_instance.sync_start_sample(sample_index, session)
                    
                    # 从 session.history 中提取 messages 用于后续处理
                    messages = []
                    for item in session.history:
                        if hasattr(item, 'root') and isinstance(item.root, dict):
                            msg = item.root
                            if msg.get("role") in ["system", "user", "assistant"]:
                                messages.append(msg)
                        elif isinstance(item, dict):
                            if item.get("role") in ["system", "user", "assistant"]:
                                messages.append(item)
                    
                    # 从 history 中提取 reward（用于 previous_sample_utilization 等记忆机制）
                    # 对于 locomo 任务，reward 根据 llm_score 定义
                    reward = 0  # 默认 reward 为 0
                    for item in session.history:
                        if hasattr(item, 'root'):
                            # RootModel 类型，检查 root 是否是 RewardHistoryItem
                            if hasattr(item.root, 'reward'):
                                reward_item = item.root
                                # 优先从 metrics 中的 llm_score 获取 reward
                                if hasattr(reward_item, 'metrics') and isinstance(reward_item.metrics, dict):
                                    llm_score = reward_item.metrics.get("llm_score")
                                    if llm_score is not None:
                                        reward = float(llm_score)  # llm_score 是 0 或 1
                                        break
                                # 如果没有 metrics，使用 reward 字段
                                reward = reward_item.reward
                                break
                        elif isinstance(item, dict) and "reward" in item:
                            # 如果是字典，检查是否有 metrics
                            if "metrics" in item and isinstance(item["metrics"], dict):
                                llm_score = item["metrics"].get("llm_score")
                                if llm_score is not None:
                                    reward = float(llm_score)
                                    break
                            reward = item["reward"]
                            break
                        elif hasattr(item, 'reward'):
                            # 直接是 RewardHistoryItem 实例
                            # 优先从 metrics 中的 llm_score 获取 reward
                            if hasattr(item, 'metrics') and isinstance(item.metrics, dict):
                                llm_score = item.metrics.get("llm_score")
                                if llm_score is not None:
                                    reward = float(llm_score)
                                    break
                            reward = item.reward
                            break
                    
                    # 如果从 history 中没有找到，尝试从 task_result.result 中的 metrics 获取
                    if reward == 0 and isinstance(task_result.result, dict):
                        metrics = task_result.result.get("metrics")
                        if isinstance(metrics, dict):
                            llm_score = metrics.get("llm_score")
                            if llm_score is not None:
                                reward = float(llm_score)
                    
                    # 使用 task_result 作为 result（添加 reward 字段以便记忆机制识别）
                    result = {
                        "status": task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status),
                        "result": task_result.result,
                        "reward": reward,  # 添加 reward 字段，便于 previous_sample_utilization 等记忆机制识别
                    }
                    
                    # 更新 memory（使用 session.history）
                    # 根据 training_mode 决定是否更新（transfer 和 replay 模式）
                    history = session.history
                    should_update_memory_locomo = True
                    if training_mode == "transfer":
                        transfer_task = exp_cfg.experiment.get("transfer_task")
                        transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                        # 只有在跨任务迁移（transfer_task != transfer_after_task）且当前任务是 transfer_after_task 时，才不更新记忆
                        # 前向迁移（transfer_task == transfer_after_task）时，所有样本都应该更新记忆
                        if transfer_task != transfer_after_task and task_name == transfer_after_task:
                            should_update_memory_locomo = False
                    elif training_mode == "replay":
                        if is_replay_test:
                            should_update_memory_locomo = False
                    
                    if should_update_memory_locomo:
                        if isinstance(memory, dict):
                            for agent_mem in memory.values():
                                agent_mem.update_memory(task_name, history, result)
                        else:
                            memory.update_memory(task_name, history, result)

                    # Replay 模式：立即测试（学习后立即测试该样本）
                    if training_mode == "replay" and should_update_memory_locomo and not is_replay_test:
                        print(f"  [Immediate Test] Testing sample {sample_index} immediately after learning...")
                        try:
                            # 获取 agent_name
                            immediate_test_agent_name = "unknown"
                            if isinstance(execution_engine, SingleAgentExecutionEngine):
                                immediate_test_agent_name = execution_engine.config.agent_name

                            # 创建新的 session 用于立即测试
                            immediate_test_session = LocomoSessionWrapper(sample_index, llm_agent, memory_for_enhance, task_name, locomo_task_instance, training_mode)

                            # 执行测试（仅enhance，不update）
                            test_task_result = locomo_task_instance.sync_start_sample(sample_index, immediate_test_session)

                            # 提取 messages
                            test_messages = []
                            for item in immediate_test_session.history:
                                if hasattr(item, 'root') and isinstance(item.root, dict):
                                    msg = item.root
                                    if msg.get("role") in ["system", "user", "assistant"]:
                                        test_messages.append(msg)
                                elif isinstance(item, dict) and item.get("role") in ["system", "user", "assistant"]:
                                    test_messages.append(item)

                            # 提取 reward
                            test_reward = 0
                            for item in immediate_test_session.history:
                                if hasattr(item, 'root') and hasattr(item.root, 'reward'):
                                    reward_item = item.root
                                    if hasattr(reward_item, 'metrics') and isinstance(reward_item.metrics, dict):
                                        llm_score = reward_item.metrics.get("llm_score")
                                        if llm_score is not None:
                                            test_reward = float(llm_score)
                                            break
                                    test_reward = reward_item.reward
                                    break
                                elif isinstance(item, dict) and "reward" in item:
                                    if "metrics" in item and isinstance(item["metrics"], dict):
                                        llm_score = item["metrics"].get("llm_score")
                                        if llm_score is not None:
                                            test_reward = float(llm_score)
                                            break
                                    test_reward = item["reward"]
                                    break

                            # 如果从 history 中没有找到，尝试从 task_result 中获取
                            if test_reward == 0 and isinstance(test_task_result.result, dict):
                                metrics = test_task_result.result.get("metrics")
                                if isinstance(metrics, dict):
                                    llm_score = metrics.get("llm_score")
                                    if llm_score is not None:
                                        test_reward = float(llm_score)

                            # 构建 test_result
                            test_result = {
                                "status": test_task_result.status.value if hasattr(test_task_result.status, 'value') else str(test_task_result.status),
                                "result": test_task_result.result,
                                "reward": test_reward,
                            }

                            # 转换为可序列化格式
                            test_serializable_history = []
                            for item in immediate_test_session.history:
                                if hasattr(item, 'root'):
                                    test_serializable_history.append(item.root)
                                elif hasattr(item, 'model_dump'):
                                    test_serializable_history.append(item.model_dump(exclude_none=True))
                                elif isinstance(item, dict):
                                    test_serializable_history.append(item)
                                else:
                                    test_serializable_history.append(str(item))

                            # 保存立即测试结果到 test/ 目录（与 replay1/ 平级）
                            immediate_test_dir = ensure_output_dir(train_output_root / "test")
                            immediate_test_task_dir = ensure_output_dir(immediate_test_dir / task_name)
                            immediate_test_out_path = immediate_test_task_dir / f"{sample_index}.json"

                            with immediate_test_out_path.open("w", encoding="utf-8") as f:
                                json.dump({
                                    "task": task_name,
                                    "index": sample_index,
                                    "split": "immediate_test",
                                    "status": test_result["status"],
                                    "result": test_result["result"],
                                    "history": test_serializable_history,
                                    "agent_name": immediate_test_agent_name,
                                }, f, indent=2, ensure_ascii=False)

                            # 记录到 execution_order_test（用于test目录的execution_order.json）
                            if task_name not in execution_order_test:
                                execution_order_test[task_name] = []
                            execution_order_test[task_name].append({
                                "task": task_name,
                                "index": sample_index,
                                "split": "immediate_test",
                                "execution_order": len(execution_order_test[task_name]) + 1,
                                "timestamp": time.time(),
                                "status": test_result["status"],
                            })

                            print(f"  [Immediate Test] -> Completed: status={test_result['status']}, reward={test_reward:.2f}")
                        except Exception as e:
                            print(f"  [Immediate Test] -> ERROR: {str(e)}")
                            logging.error(f"Immediate test failed for {task_name}[{sample_index}]: {str(e)}", exc_info=True)

                    # 保存结果
                    # 将 history 转换为可序列化的格式
                    serializable_history = []
                    for item in history:
                        if hasattr(item, 'root'):
                            # RootModel 类型，获取 root 值
                            serializable_history.append(item.root)
                        elif hasattr(item, 'model_dump'):
                            # Pydantic 模型，转换为字典
                            # 使用 exclude_none=True 排除 None 值（如 score=None）
                            serializable_history.append(item.model_dump(exclude_none=True))
                        elif isinstance(item, dict):
                            serializable_history.append(item)
                        else:
                            # 其他类型，尝试转换为字符串
                            serializable_history.append(str(item))
                    
                    # 获取 agent_name
                    agent_name = "unknown"
                    if isinstance(execution_engine, SingleAgentExecutionEngine):
                        agent_name = execution_engine.config.agent_name

                    # 确定 split：transfer 模式的 transfer_after_task 和 replay 模式的测试样本为 "test"
                    split = "train"
                    if training_mode == "transfer":
                        transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                        if task_name == transfer_after_task:
                            split = "test"
                    elif training_mode == "replay":
                        if is_replay_test:
                            split = "test"
                        else:
                            # 训练样本：添加到已学习列表
                            learned_samples_in_replay.append(sample_index)
                            # 确定当前 replay_id（根据已学习的样本数量）
                            if replay_info:
                                for rid, info in replay_info.items():
                                    if sample_index in info["train"]:
                                        # 找到包含当前样本的 replay，取最大的 replay_id
                                        current_replay_id = max(current_replay_id, rid)
                    
                    # Replay 模式：保存到对应的 replay 文件夹
                    if training_mode == "replay" and replay_info:
                        if is_replay_test:
                            # 测试样本：保存到当前 replay 的 test 文件夹
                            # 确定当前 replay_id（根据测试样本所属的 replay）
                            for rid, info in replay_info.items():
                                if sample_index in info["test"]:
                                    current_replay_id = rid
                                    break
                            
                            replay_dir = ensure_output_dir(train_output_root / f"replay{current_replay_id}" / "test")
                            task_dir = ensure_output_dir(replay_dir / task_name)
                            out_path = task_dir / f"{sample_index}.json"
                        else:
                            # 训练样本：保存到当前及之后所有 replay 的 train 文件夹
                            # 找到所有包含当前样本的 replay（当前及之后的所有 replay）
                            target_replays = []
                            for rid, info in replay_info.items():
                                if sample_index in info["train"]:
                                    target_replays.append(rid)
                            
                            # 保存到所有目标 replay 的 train 文件夹
                            for rid in target_replays:
                                replay_dir = ensure_output_dir(train_output_root / f"replay{rid}" / "train")
                                task_dir = ensure_output_dir(replay_dir / task_name)
                                out_path = task_dir / f"{sample_index}.json"
                                with out_path.open("w", encoding="utf-8") as f:
                                    json.dump({
                                        "task": task_name,
                                        "index": sample_index,
                                        "split": split,
                                        "status": result["status"],
                                        "result": result["result"],
                                        "history": serializable_history,
                                        "agent_name": agent_name,
                                    }, f, indent=2, ensure_ascii=False)
                            
                            # 记录执行顺序（只记录一次，使用第一个 replay）
                            if target_replays:
                                if task_name not in execution_order_train:
                                    execution_order_train[task_name] = []
                                execution_order_train[task_name].append({
                                    "task": task_name,
                                    "index": sample_index,
                                    "split": split,
                                    "execution_order": len(execution_order_train[task_name]) + 1,
                                    "timestamp": time.time(),
                                    "status": result["status"],
                                })
                            
                            print(f"  -> Completed: status={result['status']} (saved to replay{target_replays[0]}-{target_replays[-1]}/train)")
                            continue  # 跳过后续的保存逻辑
                    else:
                        # 非 replay 模式或 replay_info 为 None：使用原有逻辑
                        task_dir = ensure_output_dir(train_output_root / task_name)
                        out_path = task_dir / f"{sample_index}.json"
                    
                    # 保存结果（replay 模式的测试样本，或非 replay 模式）
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump({
                            "task": task_name,
                            "index": sample_index,
                            "split": split,
                            "status": result["status"],
                            "result": result["result"],
                            "history": serializable_history,
                            "agent_name": agent_name,
                        }, f, indent=2, ensure_ascii=False)
                    
                    # 记录执行顺序（根据 split 选择对应的 execution_order 字典）
                    # Replay 模式的测试样本需要单独记录
                    if training_mode == "replay" and is_replay_test:
                        # Replay 模式的测试样本：记录到当前 replay 的执行顺序
                        if task_name not in execution_order_test:
                            execution_order_test[task_name] = []
                        execution_order_test[task_name].append({
                            "task": task_name,
                            "index": sample_index,
                            "split": split,
                            "execution_order": len(execution_order_test[task_name]) + 1,
                            "timestamp": time.time(),
                            "status": result["status"],
                            "replay_id": current_replay_id,
                        })
                    elif split == "test":
                        if task_name not in execution_order_test:
                            execution_order_test[task_name] = []
                        execution_order_test[task_name].append({
                            "task": task_name,
                            "index": sample_index,
                            "split": split,
                            "execution_order": len(execution_order_test[task_name]) + 1,
                            "timestamp": time.time(),
                            "status": result["status"],
                        })
                    else:
                        if task_name not in execution_order_train:
                            execution_order_train[task_name] = []
                        execution_order_train[task_name].append({
                            "task": task_name,
                            "index": sample_index,
                            "split": split,
                            "execution_order": len(execution_order_train[task_name]) + 1,
                            "timestamp": time.time(),
                            "status": result["status"],
                        })

                    print(f"  -> Completed: status={result['status']}")

                    # 前向迁移测试（transfer mode with same task）
                    if training_mode == "transfer":
                        transfer_task = exp_cfg.experiment.get("transfer_task")
                        transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                        forward_transfer_num = exp_cfg.experiment.get("forward_transfer_num")

                        # DEBUG: 输出关键变量
                        print(f"  [Forward Transfer DEBUG] transfer_task={transfer_task}, transfer_after_task={transfer_after_task}, forward_transfer_num={forward_transfer_num}, should_update={should_update_memory_locomo}")

                        # 只有当 transfer_task == transfer_after_task 且更新了记忆时，才进行前向迁移测试
                        if transfer_task == transfer_after_task and should_update_memory_locomo and forward_transfer_num:
                            print(f"  [Forward Transfer] Checking forward test after training on {task_name}[{sample_index}] (forward_num={forward_transfer_num})")
                            # 对于 locomo 任务，也从 schedule 中往后数第 N 个样本（与 db 任务一致）
                            # forward_transfer_num 指的是次序上的后 N 个样本，不是 index 上的 +N
                            current_position = idx - 1  # idx 从 1 开始，转换为 0-based index

                            # 从当前位置往后查找第 N 个非 session 样本
                            forward_test_target = None
                            count = 0
                            for i in range(current_position + 1, len(train_schedule)):
                                future_task_name, future_sample_index = train_schedule[i]

                                # 跳过 session injection marker
                                if future_task_name == SESSION_INJECTION_MARKER:
                                    continue

                                count += 1
                                if count == forward_transfer_num:
                                    forward_test_target = (future_task_name, future_sample_index)
                                    print(f"  [Forward Transfer] Found target at schedule position {i}: {future_task_name}[{future_sample_index}]")
                                    break

                            if not forward_test_target:
                                print(f"  [Forward Transfer] Skipped - not enough future samples (found {count}, need {forward_transfer_num})")

                            # 如果找到了目标样本，执行前向测试
                            if forward_test_target:
                                test_task_name, test_sample_index = forward_test_target
                                print(f"  [Forward Transfer Test] Testing future sample: {test_task_name}[{test_sample_index}] (forward_num={forward_transfer_num})")

                                try:
                                    # 对于 locomo 任务，使用 LocomoSessionWrapper 执行前向测试
                                    if is_locomo_task(test_task_name) and locomo_task_instance is not None and locomo_task_name == test_task_name:
                                        # 创建包装的 Session（只 enhance，不 update）
                                        test_session = LocomoSessionWrapper(test_sample_index, llm_agent, memory_for_enhance, test_task_name, locomo_task_instance, training_mode)

                                        # 直接调用任务实例的 sync_start_sample
                                        test_task_result = locomo_task_instance.sync_start_sample(test_sample_index, test_session)

                                        # 从 test_session.history 中提取 messages 用于后续处理
                                        test_messages = []
                                        for item in test_session.history:
                                            if hasattr(item, 'root') and isinstance(item.root, dict):
                                                msg = item.root
                                                if msg.get("role") in ["system", "user", "assistant"]:
                                                    test_messages.append(msg)
                                            elif isinstance(item, dict):
                                                if item.get("role") in ["system", "user", "assistant"]:
                                                    test_messages.append(item)

                                        # 从 history 中提取 reward
                                        test_reward = 0  # 默认 reward 为 0
                                        for item in test_session.history:
                                            if hasattr(item, 'root'):
                                                # RootModel 类型，检查 root 是否是 RewardHistoryItem
                                                if hasattr(item.root, 'reward'):
                                                    reward_item = item.root
                                                    # 优先从 metrics 中的 llm_score 获取 reward
                                                    if hasattr(reward_item, 'metrics') and isinstance(reward_item.metrics, dict):
                                                        llm_score = reward_item.metrics.get("llm_score")
                                                        if llm_score is not None:
                                                            test_reward = float(llm_score)  # llm_score 是 0 或 1
                                                            break
                                                    # 如果没有 metrics，使用 reward 字段
                                                    test_reward = reward_item.reward
                                                    break
                                            elif isinstance(item, dict) and "reward" in item:
                                                # 如果是字典，检查是否有 metrics
                                                if "metrics" in item and isinstance(item["metrics"], dict):
                                                    llm_score = item["metrics"].get("llm_score")
                                                    if llm_score is not None:
                                                        test_reward = float(llm_score)
                                                        break
                                                test_reward = item["reward"]
                                                break
                                            elif hasattr(item, 'reward'):
                                                # 直接是 RewardHistoryItem 实例
                                                # 优先从 metrics 中的 llm_score 获取 reward
                                                if hasattr(item, 'metrics') and isinstance(item.metrics, dict):
                                                    llm_score = item.metrics.get("llm_score")
                                                    if llm_score is not None:
                                                        test_reward = float(llm_score)
                                                        break
                                                test_reward = item.reward
                                                break

                                        # 如果从 history 中没有找到，尝试从 test_task_result.result 中的 metrics 获取
                                        if test_reward == 0 and isinstance(test_task_result.result, dict):
                                            metrics = test_task_result.result.get("metrics")
                                            if isinstance(metrics, dict):
                                                llm_score = metrics.get("llm_score")
                                                if llm_score is not None:
                                                    test_reward = float(llm_score)

                                        # 使用 test_task_result 作为 result
                                        test_result = {
                                            "status": test_task_result.status.value if hasattr(test_task_result.status, 'value') else str(test_task_result.status),
                                            "result": test_task_result.result,
                                            "reward": test_reward,
                                        }

                                        # 将 history 转换为可序列化的格式
                                        test_history = []
                                        for item in test_session.history:
                                            if hasattr(item, 'root'):
                                                # RootModel 类型，获取 root 值
                                                test_history.append(item.root)
                                            elif hasattr(item, 'model_dump'):
                                                # Pydantic 模型，转换为字典
                                                test_history.append(item.model_dump(exclude_none=True))
                                            elif isinstance(item, dict):
                                                test_history.append(item)
                                            else:
                                                # 其他类型，尝试转换为字符串
                                                test_history.append(str(item))
                                    else:
                                        # 非 locomo 任务：这种情况不应该出现在 locomo 分支中
                                        print(f"  [Forward Transfer Test] -> WARNING: Non-locomo task in locomo branch")
                                        continue

                                    # 保存前向测试结果到单独的目录
                                    forward_test_dir = ensure_output_dir(base_output_root / "forward_transfer_test" / test_task_name)
                                    test_out_path = forward_test_dir / f"train{sample_index}_test{test_sample_index}.json"

                                    # 确保 agent_name 在结果中
                                    test_agent_name = "unknown"
                                    if isinstance(execution_engine, SingleAgentExecutionEngine):
                                        test_agent_name = execution_engine.config.agent_name

                                    test_output_data = {
                                        "task": test_task_name,
                                        "index": test_sample_index,
                                        "split": "forward_test",
                                        "trained_on_index": sample_index,  # 记录是在哪个样本训练后测试的
                                        "forward_num": forward_transfer_num,
                                        "history": test_history,
                                        "result": test_result,
                                        "agent_name": test_agent_name,
                                    }

                                    with test_out_path.open("w", encoding="utf-8") as f:
                                        json.dump(test_output_data, f, ensure_ascii=False, indent=2)

                                    test_status = test_result.get("status", "unknown") if isinstance(test_result, dict) else "unknown"
                                    print(f"  [Forward Transfer Test] -> status={test_status}, saved to {test_out_path.relative_to(ROOT_DIR)}")

                                    # 记录前向迁移测试的执行顺序
                                    if test_task_name not in execution_order_forward_test:
                                        execution_order_forward_test[test_task_name] = []
                                    execution_order_forward_test[test_task_name].append({
                                        "task": test_task_name,
                                        "index": test_sample_index,
                                        "split": "forward_test",
                                        "trained_on_index": sample_index,
                                        "forward_num": forward_transfer_num,
                                        "execution_order": len(execution_order_forward_test[test_task_name]) + 1,
                                        "timestamp": time.time(),
                                        "status": test_status,
                                    })

                                except Exception as e:
                                    print(f"  [Forward Transfer Test] -> ERROR: {str(e)}")
                                    logging.error(f"Forward transfer test failed for {test_task_name}[{test_sample_index}]: {str(e)}", exc_info=True)

                    continue  # 跳过后续的后端处理
                
                # 6.1 调用 /start_sample，获取 session_id + 初始 messages/tools
                session_id, messages, tools = backend.start_sample(task_name, sample_index)
                print(f"  -> backend returned session_id={session_id}, messages={len(messages)}, tools={len(tools)}")

                # 6.1.1 对于 kg 任务，过滤掉演示模板，只保留 system 和最后一个 user 消息
                if task_name.startswith("kg-") or "kg" in task_name.lower():
                    original_count = len(messages)
                    filtered_messages = []
                    user_messages = [msg for msg in messages if msg.get("role") == "user"]
                    
                    # 保留第一个 system 消息
                    for msg in messages:
                        if msg.get("role") == "system":
                            filtered_messages.append(msg)
                            break
                    
                    # 保留最后一个 user 消息（真正的问题）
                    if user_messages:
                        filtered_messages.append(user_messages[-1])
                    
                    if filtered_messages:
                        messages = filtered_messages
                        print(f"  -> Filtered kg task messages: {len(messages)} messages (removed {original_count - len(messages)} demo template messages)")

                # 6.2 通过 memory 机制改写 messages
                # offline 模式：使用 zero_shot（不使用记忆）；online 模式：使用配置的记忆机制
                # Replay 模式：test 样本仅 enhance，不 update，直接使用当前累积的记忆

                # 直接使用当前的 memory_for_enhance（测试阶段不更新记忆，所以使用当前累积的记忆即可）
                test_memory_for_enhance = memory_for_enhance

                # single_agent
                enhanced_messages = test_memory_for_enhance.use_memory(task_name, messages)

                # 6.3 通过 execution engine 执行
                history, result = execution_engine.run_sample(
                    task=task_name,
                    index=sample_index,
                    session_id=session_id,
                    messages=enhanced_messages,
                    tools=tools,
                    agent_pool=llm_agent,
                    backend_client=backend,
                )

                # 6.3.1 确保 result 中记录 agent_name（对于 single_agent 也记录）
                if isinstance(result, dict):
                    if isinstance(execution_engine, SingleAgentExecutionEngine):
                        result["agent_name"] = execution_engine.config.agent_name

                # 6.4 更新记忆（根据 training_mode 决定是否更新）
                # Transfer 模式：
                #   - 跨任务迁移（transfer_task != transfer_after_task）：transfer_task 更新，transfer_after_task 不更新
                #   - 前向迁移（transfer_task == transfer_after_task）：所有样本都更新
                # Replay 模式：训练样本更新，测试样本不更新（通过 schedule 中的标记判断）
                should_update_memory = True
                if training_mode == "transfer":
                    transfer_task = exp_cfg.experiment.get("transfer_task")
                    transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                    # 只有在跨任务迁移且当前任务是 transfer_after_task 时，才不更新记忆
                    # 前向迁移时，所有样本都应该更新记忆
                    if transfer_task != transfer_after_task and task_name == transfer_after_task:
                        # transfer_after_task 是测试任务，不更新记忆
                        should_update_memory = False
                    elif task_name == transfer_task:
                        # transfer_task 是训练任务，更新记忆
                        should_update_memory = True
                elif training_mode == "replay":
                    # replay 模式：通过 is_replay_test 标志来判断
                    if is_replay_test:
                        # 测试样本，不更新记忆
                        should_update_memory = False
                    else:
                        # 训练样本，更新记忆
                        should_update_memory = True

                if should_update_memory:
                    # single_agent: 更新 memory
                    memory.update_memory(task_name, history, result)

                # Replay 模式：立即测试（学习后立即测试该样本）
                if training_mode == "replay" and should_update_memory and not is_replay_test:
                    print(f"  [Immediate Test] Testing sample {sample_index} immediately after learning...")
                    try:
                        # 获取初始messages（重新调用backend获取干净的初始状态）
                        test_session_id, test_messages, test_tools = backend.start_sample(task_name, sample_index)

                        # KG任务：过滤消息
                        if task_name.startswith("kg-") or "kg" in task_name.lower():
                            test_user_messages = [msg for msg in test_messages if msg.get("role") == "user"]
                            test_filtered_messages = []
                            for msg in test_messages:
                                if msg.get("role") == "system":
                                    test_filtered_messages.append(msg)
                                    break
                            if test_user_messages:
                                test_filtered_messages.append(test_user_messages[-1])
                            if test_filtered_messages:
                                test_messages = test_filtered_messages

                        # 使用memory_for_enhance（仅enhance，不update）
                        test_enhanced_messages = memory_for_enhance.use_memory(task_name, test_messages)

                        # 执行测试
                        test_history, test_result = execution_engine.run_sample(
                            task=task_name,
                            index=sample_index,
                            session_id=test_session_id,
                            messages=test_enhanced_messages,
                            tools=test_tools,
                            agent_pool=llm_agent,
                            backend_client=backend,
                        )

                        # 确保 agent_name 在结果中
                        if isinstance(test_result, dict):
                            if isinstance(execution_engine, SingleAgentExecutionEngine):
                                test_result["agent_name"] = execution_engine.config.agent_name

                        # 转换为可序列化格式
                        test_serializable_history = []
                        for msg in test_history:
                            if isinstance(msg, dict):
                                test_serializable_history.append(msg)
                            elif hasattr(msg, 'model_dump'):
                                test_serializable_history.append(msg.model_dump())
                            else:
                                test_serializable_history.append(str(msg))

                        # 获取 agent_name
                        test_agent_name = "unknown"
                        if isinstance(execution_engine, SingleAgentExecutionEngine):
                            test_agent_name = execution_engine.config.agent_name

                        # 保存立即测试结果到 test/ 目录（与 replay1/ 平级）
                        immediate_test_dir = ensure_output_dir(train_output_root / "test")
                        immediate_test_task_dir = ensure_output_dir(immediate_test_dir / task_name)
                        immediate_test_out_path = immediate_test_task_dir / f"{sample_index}.json"

                        with immediate_test_out_path.open("w", encoding="utf-8") as f:
                            json.dump({
                                "task": task_name,
                                "index": sample_index,
                                "split": "immediate_test",
                                "history": test_serializable_history,
                                "result": test_result,
                                "agent_name": test_agent_name,
                            }, f, ensure_ascii=False, indent=2)

                        # 记录到 execution_order_test（用于test目录的execution_order.json）
                        if task_name not in execution_order_test:
                            execution_order_test[task_name] = []
                        test_status = test_result.get("status", "unknown") if isinstance(test_result, dict) else "unknown"
                        execution_order_test[task_name].append({
                            "task": task_name,
                            "index": sample_index,
                            "split": "immediate_test",
                            "execution_order": len(execution_order_test[task_name]) + 1,
                            "timestamp": time.time(),
                            "status": test_status,
                        })

                        print(f"  [Immediate Test] -> Completed: status={test_status}")
                    except Exception as e:
                        print(f"  [Immediate Test] -> ERROR: {str(e)}")
                        logging.error(f"Immediate test failed for {task_name}[{sample_index}]: {str(e)}", exc_info=True)

                # 6.4.1 前向迁移测试（transfer mode with same task）
                if training_mode == "transfer":
                    transfer_task = exp_cfg.experiment.get("transfer_task")
                    transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                    forward_transfer_num = exp_cfg.experiment.get("forward_transfer_num")

                    # 只有当 transfer_task == transfer_after_task 且更新了记忆时，才进行前向迁移测试
                    if transfer_task == transfer_after_task and should_update_memory and forward_transfer_num:
                        print(f"  [Forward Transfer] Checking forward test after training on {task_name}[{sample_index}] (forward_num={forward_transfer_num})")
                        # 计算前向测试的目标索引
                        # 需要从 train_schedule 中找到当前样本的位置，然后往后数 forward_transfer_num 个非 session 的样本
                        current_position = idx - 1  # idx 从 1 开始，转换为 0-based index

                        # 从当前位置往后查找第 N 个非 session 样本
                        forward_test_target = None
                        count = 0
                        for i in range(current_position + 1, len(train_schedule)):
                            future_task_name, future_sample_index = train_schedule[i]

                            # 跳过 session injection marker
                            if future_task_name == SESSION_INJECTION_MARKER:
                                continue

                            count += 1
                            if count == forward_transfer_num:
                                forward_test_target = (future_task_name, future_sample_index)
                                print(f"  [Forward Transfer] Found target at schedule position {i}: {future_task_name}[{future_sample_index}]")
                                break

                        if not forward_test_target:
                            print(f"  [Forward Transfer] Skipped - not enough future samples (found {count}, need {forward_transfer_num})")

                        # 如果找到了目标样本，执行前向测试
                        if forward_test_target:
                            test_task_name, test_sample_index = forward_test_target
                            print(f"  [Forward Transfer Test] Testing future sample: {test_task_name}[{test_sample_index}] (forward_num={forward_transfer_num})")

                            try:
                                # 对于 db 等 system memory 任务，需要手动调用 backend
                                # 1. 调用 backend.start_sample 获取 session_id, messages, tools
                                test_session_id, test_messages, test_tools = backend.start_sample(test_task_name, test_sample_index)

                                # 2. 通过 memory 机制增强 messages（只 enhance，不 update）
                                test_enhanced_messages = memory_for_enhance.use_memory(test_task_name, test_messages)

                                # 3. 通过 execution engine 执行
                                test_history, test_result = execution_engine.run_sample(
                                    task=test_task_name,
                                    index=test_sample_index,
                                    session_id=test_session_id,
                                    messages=test_enhanced_messages,
                                    tools=test_tools,
                                    agent_pool=llm_agent,
                                    backend_client=backend,
                                )

                                # 保存前向测试结果到单独的目录
                                forward_test_dir = ensure_output_dir(base_output_root / "forward_transfer_test" / test_task_name)
                                test_out_path = forward_test_dir / f"train{sample_index}_test{test_sample_index}.json"

                                # 确保 agent_name 在结果中
                                test_agent_name = None
                                if isinstance(test_result, dict):
                                    if isinstance(execution_engine, SingleAgentExecutionEngine):
                                        test_result["agent_name"] = execution_engine.config.agent_name
                                    test_agent_name = test_result.get("agent_name")

                                test_output_data = {
                                    "task": test_task_name,
                                    "index": test_sample_index,
                                    "split": "forward_test",
                                    "trained_on_index": sample_index,  # 记录是在哪个样本训练后测试的
                                    "forward_num": forward_transfer_num,
                                    "history": test_history,
                                    "result": test_result,
                                }
                                if test_agent_name:
                                    test_output_data["agent_name"] = test_agent_name

                                with test_out_path.open("w", encoding="utf-8") as f:
                                    json.dump(test_output_data, f, ensure_ascii=False, indent=2)

                                test_status = test_result.get("status", "unknown") if isinstance(test_result, dict) else "unknown"
                                print(f"  [Forward Transfer Test] -> status={test_status}, saved to {test_out_path.relative_to(ROOT_DIR)}")

                                # 记录前向迁移测试的执行顺序
                                if test_task_name not in execution_order_forward_test:
                                    execution_order_forward_test[test_task_name] = []
                                execution_order_forward_test[test_task_name].append({
                                    "task": test_task_name,
                                    "index": test_sample_index,
                                    "split": "forward_test",
                                    "trained_on_index": sample_index,
                                    "forward_num": forward_transfer_num,
                                    "execution_order": len(execution_order_forward_test[test_task_name]) + 1,
                                    "timestamp": time.time(),
                                    "status": test_status,
                                })

                            except Exception as e:
                                print(f"  [Forward Transfer Test] -> ERROR: {str(e)}")
                                logging.error(f"Forward transfer test failed for {test_task_name}[{test_sample_index}]: {str(e)}", exc_info=True)

                # 6.5 落盘（根据 training_mode 决定 split 和目录）
                # 确保 agent_name 在顶层（从 result 中提取，如果存在）
                agent_name = None
                if isinstance(result, dict):
                    agent_name = result.get("agent_name")
                
                # 确定 split：transfer 模式的 transfer_after_task 和 replay 模式的测试样本为 "test"
                split = "train"
                if training_mode == "transfer":
                    transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                    if task_name == transfer_after_task:
                        split = "test"
                elif training_mode == "replay":
                    if is_replay_test:
                        split = "test"
                    else:
                        # 训练样本：添加到已学习列表
                        learned_samples_in_replay.append(sample_index)
                        # 确定当前 replay_id（根据已学习的样本数量）
                        if replay_info:
                            for rid, info in replay_info.items():
                                if sample_index in info["train"]:
                                    # 找到包含当前样本的 replay，取最大的 replay_id
                                    current_replay_id = max(current_replay_id, rid)
                        
                        # Replay 模式：检查是否完成了某个 replay 的所有训练样本
                        # 完成后更新 current_replay_id_for_test，用于确定测试样本应该保存到哪个 replay
                        if replay_info:
                            # 找到当前样本所属的最大 replay_id
                            current_sample_replay_id = 0
                            for rid, info in replay_info.items():
                                if sample_index in info["train"]:
                                    current_sample_replay_id = max(current_sample_replay_id, rid)
                            
                            # 检查当前样本所属的 replay 的所有训练样本是否都已完成
                            if current_sample_replay_id > 0:
                                info = replay_info[current_sample_replay_id]
                                train_samples = set(info["train"])
                                # 检查 learned_samples_in_replay 是否包含了该 replay 的所有训练样本
                                if train_samples.issubset(set(learned_samples_in_replay)):
                                    # 该 replay 的所有训练样本都已完成，更新 current_replay_id_for_test
                                    # 下一个测试阶段应该保存到这个 replay 的 test 文件夹
                                    if current_replay_id_for_test < current_sample_replay_id:
                                        current_replay_id_for_test = current_sample_replay_id
                                        print(f"[Replay] Completed all training samples for replay{current_sample_replay_id}, current_replay_id_for_test={current_replay_id_for_test}")
                
                # Replay 模式：保存到对应的 replay 文件夹
                if training_mode == "replay" and replay_info:
                    if is_replay_test:
                        # 测试样本：只保存到当前 replay 的 test 文件夹（使用 current_replay_id）
                        # 注意：一个测试样本可能出现在多个 replay 的 test 列表中，但执行时应该只保存到当前 replay
                        if current_replay_id > 0:
                            replay_dir = ensure_output_dir(train_output_root / f"replay{current_replay_id}" / "test")
                            task_dir = ensure_output_dir(replay_dir / task_name)
                            out_path = task_dir / f"{sample_index}.json"
                            
                            output_data = {
                                "task": task_name,
                                "index": sample_index,
                                "split": split,
                                "history": history,
                                "result": result,
                            }
                            if agent_name:
                                output_data["agent_name"] = agent_name
                            
                            with out_path.open("w", encoding="utf-8") as f:
                                json.dump(output_data, f, ensure_ascii=False, indent=2)
                            
                            # 记录执行顺序
                            if task_name not in execution_order_test:
                                execution_order_test[task_name] = []
                            execution_order_test[task_name].append({
                                "task": task_name,
                                "index": sample_index,
                                "split": split,
                                "execution_order": len(execution_order_test[task_name]) + 1,
                                "timestamp": time.time(),
                                "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
                            })
                            
                            print(f"  -> Completed: status={result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'} (saved to replay{current_replay_id}/test)")
                        else:
                            print(f"[Replay] Test sample {sample_index} has invalid current_replay_id={current_replay_id}, skipping save")
                        continue  # 跳过后续的保存逻辑
                    else:
                        # 训练样本：保存到当前及之后所有 replay 的 train 文件夹
                        # 找到所有包含当前样本的 replay（当前及之后的所有 replay）
                        target_replays = []
                        for rid, info in replay_info.items():
                            if sample_index in info["train"]:
                                target_replays.append(rid)
                        
                        # 保存到所有目标 replay 的 train 文件夹
                        for rid in target_replays:
                            replay_dir = ensure_output_dir(train_output_root / f"replay{rid}" / "train")
                            task_dir = ensure_output_dir(replay_dir / task_name)
                            out_path = task_dir / f"{sample_index}.json"
                            
                            output_data = {
                                "task": task_name,
                                "index": sample_index,
                                "split": split,
                            "history": history,
                            "result": result,
                            }
                            if agent_name:
                                output_data["agent_name"] = agent_name
                            
                            with out_path.open("w", encoding="utf-8") as f:
                                json.dump(output_data, f, ensure_ascii=False, indent=2)
                        
                        # 记录执行顺序（只记录一次，使用第一个 replay）
                        if target_replays:
                            if task_name not in execution_order_train:
                                execution_order_train[task_name] = []
                            execution_order_train[task_name].append({
                                "task": task_name,
                                "index": sample_index,
                                "split": split,
                                "execution_order": len(execution_order_train[task_name]) + 1,
                                "timestamp": time.time(),
                                "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
                            })
                        
                        print(f"  -> Completed: status={result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'} (saved to replay{target_replays[0]}-{target_replays[-1]}/train)")
                        continue  # 跳过后续的保存逻辑
                else:
                    # 非 replay 模式或 replay_info 为 None：使用原有逻辑
                    task_dir = ensure_output_dir(train_output_root / task_name)
                    out_path = task_dir / f"{sample_index}.json"
                
                # 保存结果（replay 模式的测试样本，或非 replay 模式）
                output_data = {
                    "task": task_name,
                    "index": sample_index,
                    "split": split,
                            "history": history,
                            "result": result,
                }
                if agent_name:
                    output_data["agent_name"] = agent_name
                
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                # 记录执行顺序（根据 split 选择对应的 execution_order 字典）
                # Replay 模式的测试样本需要单独记录
                if training_mode == "replay" and is_replay_test:
                    # Replay 模式的测试样本：记录到当前 replay 的执行顺序
                    if task_name not in execution_order_test:
                        execution_order_test[task_name] = []
                    execution_order_test[task_name].append({
                        "task": task_name,
                        "index": sample_index,
                        "split": split,
                        "execution_order": len(execution_order_test[task_name]) + 1,
                        "timestamp": time.time(),
                        "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
                        "replay_id": current_replay_id,
                    })
                elif split == "test":
                    if task_name not in execution_order_test:
                        execution_order_test[task_name] = []
                    execution_order_test[task_name].append({
                        "task": task_name,
                        "index": sample_index,
                        "split": split,
                        "execution_order": len(execution_order_test[task_name]) + 1,
                        "timestamp": time.time(),
                        "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
                    })
                else:
                    if task_name not in execution_order_train:
                        execution_order_train[task_name] = []
                    execution_order_train[task_name].append({
                        "task": task_name,
                        "index": sample_index,
                        "split": split,
                        "execution_order": len(execution_order_train[task_name]) + 1,
                        "timestamp": time.time(),
                        "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
                    })

                # 6.6 输出 agent 信息
                agent_info = ""
                if isinstance(result, dict) and "agent_name" in result:
                    agent_name = result.get("agent_name", "unknown")
                    agent_info = f", agent={agent_name}"

                print(f"  -> saved to {out_path.relative_to(ROOT_DIR)}{agent_info}\n")

            except Exception as e:
                # 捕获所有异常，记录错误但继续处理下一个样本
                error_msg = f"  -> ERROR: Failed to process sample {sample_index} of task {task_name}: {str(e)}"
                print(error_msg)
                logging.error(error_msg, exc_info=True)
                
                # 确定 split：transfer 模式的 transfer_after_task 和 replay 模式的测试样本为 "test"
                split = "train"
                if training_mode == "transfer":
                    transfer_after_task = exp_cfg.experiment.get("transfer_after_task")
                    if task_name == transfer_after_task:
                        split = "test"
                elif training_mode == "replay":
                    if is_replay_test:
                        split = "test"
                
                # 可选：保存错误信息到文件
                task_dir = ensure_output_dir(train_output_root / task_name)
                error_path = task_dir / f"{sample_index}.error.json"
                with error_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task": task_name,
                            "index": sample_index,
                            "split": split,
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                
                # 记录执行顺序（即使出错也记录，根据 split 选择对应的 execution_order 字典）
                if split == "test":
                    if task_name not in execution_order_test:
                        execution_order_test[task_name] = []
                    execution_order_test[task_name].append({
                        "task": task_name,
                        "index": sample_index,
                        "split": split,
                        "execution_order": len(execution_order_test[task_name]) + 1,
                        "timestamp": time.time(),
                        "status": "error",
                        "error": str(e),
                    })
                else:
                    if task_name not in execution_order_train:
                        execution_order_train[task_name] = []
                    execution_order_train[task_name].append({
                        "task": task_name,
                        "index": sample_index,
                        "split": split,
                        "execution_order": len(execution_order_train[task_name]) + 1,
                        "timestamp": time.time(),
                        "status": "error",
                        "error": str(e),
                    })
                
                print(f"  -> error saved to {error_path.relative_to(ROOT_DIR)}\n")
                continue  # 跳过当前样本，继续下一个

            # 为了更保险，样本之间也稍微停顿一下（完全串行执行）
            time.sleep(1.0)

    # 7) 执行测试集样本（如果存在）
    if test_schedule:
        print(f"\n{'='*60}")
        print(f"Running TEST set: {len(test_schedule)} samples")
        print(f"{'='*60}\n")
        
        # offline 模式：对于 locomo 任务，测试阶段也需要 shuffle（如果启用）
        if training_mode == "offline":
            # 检查是否有 locomo 任务
            locomo_tasks_in_test = [name for name, _ in test_schedule if is_locomo_task(name)]
            if locomo_tasks_in_test and shuffle_enabled:
                import random
                shuffle_seed = exp_cfg.experiment.get("shuffle", {}).get("seed", None)
                if shuffle_seed is not None:
                    random.seed(shuffle_seed)
                random.shuffle(test_schedule)
                print(f"  -> Shuffled {len(test_schedule)} test QAs for locomo task (offline mode)")

        # offline 模式：测试集使用配置的 memory mechanism 进行 use_memory
        if training_mode == "offline":
            print(f"Training mode: {training_mode} -> Test set will use {exp_cfg.memory_mechanism.get('name', 'zero_shot')} for use_memory (memory enabled for testing)")
            # 重新构建 memory_for_enhance，使用配置的 memory mechanism
            # single_agent
            memory_for_enhance = memory

        for idx, (task_name, sample_index) in enumerate(test_schedule, start=1):
            # 处理 session 注入标记（用于混合调度）
            # 注意：在 offline 模式的 test 阶段，session 已经在 train 阶段注入，所以这里应该跳过
            if task_name == SESSION_INJECTION_MARKER:
                print(f"[TEST {idx}/{len(test_schedule)}] [SESSION INJECTION] Skipping session injection in test phase (offline mode)")
                continue  # 跳过执行，继续下一个样本

            if not cross_task and last_task_name is not None and task_name != last_task_name:
                memory, memory_for_enhance = build_memory_bundle()
                # offline 模式：测试集使用配置的 memory mechanism 进行 use_memory
                if training_mode == "offline":
                    memory_for_enhance = memory
                print(f"\n[Memory Reset] cross_task=False, switched task {last_task_name} -> {task_name}, memory rebuilt (test split).\n")
            last_task_name = task_name
            print(f"[TEST {idx}/{len(test_schedule)}] task={task_name}, index={sample_index}")

            try:
                # 对于 locomo 任务，直接使用任务实例，不需要后端
                if is_locomo_task(task_name) and locomo_task_instance is not None and locomo_task_name == task_name:
                    # 创建包装的 Session
                    session = LocomoSessionWrapper(sample_index, llm_agent, memory_for_enhance, task_name, locomo_task_instance, training_mode)
                    
                    # 直接调用任务实例的 sync_start_sample
                    task_result = locomo_task_instance.sync_start_sample(sample_index, session)
                    
                    # 从 session.history 中提取 messages 用于后续处理
                    messages = []
                    for item in session.history:
                        if hasattr(item, 'root') and isinstance(item.root, dict):
                            msg = item.root
                            if msg.get("role") in ["system", "user", "assistant"]:
                                messages.append(msg)
                        elif isinstance(item, dict):
                            if item.get("role") in ["system", "user", "assistant"]:
                                messages.append(item)
                    
                    # 从 history 中提取 reward（用于 previous_sample_utilization 等记忆机制）
                    # 对于 locomo 任务，reward 根据 llm_score 定义
                    reward = 0  # 默认 reward 为 0
                    for item in session.history:
                        if hasattr(item, 'root'):
                            # RootModel 类型，检查 root 是否是 RewardHistoryItem
                            if hasattr(item.root, 'reward'):
                                reward_item = item.root
                                # 优先从 metrics 中的 llm_score 获取 reward
                                if hasattr(reward_item, 'metrics') and isinstance(reward_item.metrics, dict):
                                    llm_score = reward_item.metrics.get("llm_score")
                                    if llm_score is not None:
                                        reward = float(llm_score)  # llm_score 是 0 或 1
                                        break
                                # 如果没有 metrics，使用 reward 字段
                                reward = reward_item.reward
                                break
                        elif isinstance(item, dict) and "reward" in item:
                            # 如果是字典，检查是否有 metrics
                            if "metrics" in item and isinstance(item["metrics"], dict):
                                llm_score = item["metrics"].get("llm_score")
                                if llm_score is not None:
                                    reward = float(llm_score)
                                    break
                            reward = item["reward"]
                            break
                        elif hasattr(item, 'reward'):
                            # 直接是 RewardHistoryItem 实例
                            # 优先从 metrics 中的 llm_score 获取 reward
                            if hasattr(item, 'metrics') and isinstance(item.metrics, dict):
                                llm_score = item.metrics.get("llm_score")
                                if llm_score is not None:
                                    reward = float(llm_score)
                                    break
                            reward = item.reward
                            break
                    
                    # 如果从 history 中没有找到，尝试从 task_result.result 中的 metrics 获取
                    if reward == 0 and isinstance(task_result.result, dict):
                        metrics = task_result.result.get("metrics")
                        if isinstance(metrics, dict):
                            llm_score = metrics.get("llm_score")
                            if llm_score is not None:
                                reward = float(llm_score)
                    
                    # 使用 task_result 作为 result（添加 reward 字段以便记忆机制识别）
                    result = {
                        "status": task_result.status.value if hasattr(task_result.status, 'value') else str(task_result.status),
                        "result": task_result.result,
                        "reward": reward,  # 添加 reward 字段，便于 previous_sample_utilization 等记忆机制识别
                    }
                    
                    # 测试集：在 offline 模式下不更新记忆（只评估），在 online 模式下更新记忆
                    history = session.history
                    if training_mode == "offline":
                        # offline 模式：测试集不更新记忆（只评估性能）
                        pass
                    else:
                        # online 模式：测试集也更新记忆
                        if isinstance(memory, dict):
                            for agent_mem in memory.values():
                                agent_mem.update_memory(task_name, history, result)
                        else:
                            memory.update_memory(task_name, history, result)
                    
                    # 保存结果
                    # 将 history 转换为可序列化的格式
                    serializable_history = []
                    for item in history:
                        if hasattr(item, 'root'):
                            # RootModel 类型，获取 root 值
                            serializable_history.append(item.root)
                        elif hasattr(item, 'model_dump'):
                            # Pydantic 模型，转换为字典
                            # 使用 exclude_none=True 排除 None 值（如 score=None）
                            serializable_history.append(item.model_dump(exclude_none=True))
                        elif isinstance(item, dict):
                            serializable_history.append(item)
                        else:
                            # 其他类型，尝试转换为字符串
                            serializable_history.append(str(item))

                    # 获取 agent_name
                    agent_name = "unknown"
                    if isinstance(execution_engine, SingleAgentExecutionEngine):
                        agent_name = execution_engine.config.agent_name

                    task_dir = ensure_output_dir(test_output_root / task_name)
                    out_path = task_dir / f"{sample_index}.json"
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump({
                            "task": task_name,
                            "index": sample_index,
                            "split": "test",
                            "status": result["status"],
                            "result": result["result"],
                            "history": serializable_history,
                            "agent_name": agent_name,
                        }, f, indent=2, ensure_ascii=False)
                    
                    # 记录执行顺序
                    if task_name not in execution_order_test:
                        execution_order_test[task_name] = []
                    execution_order_test[task_name].append({
                        "task": task_name,
                        "index": sample_index,
                        "split": "test",
                        "execution_order": len(execution_order_test[task_name]) + 1,
                        "timestamp": time.time(),
                        "status": result["status"],
                    })
                    
                    print(f"  -> Completed: status={result['status']}")
                    continue  # 跳过后续的后端处理
                
                # 7.1 调用 /start_sample，获取 session_id + 初始 messages/tools
                session_id, messages, tools = backend.start_sample(task_name, sample_index)
                print(f"  -> backend returned session_id={session_id}, messages={len(messages)}, tools={len(tools)}")

                # 7.1.1 对于 kg 任务，过滤掉演示模板，只保留 system 和最后一个 user 消息
                if task_name.startswith("kg-") or "kg" in task_name.lower():
                    original_count = len(messages)
                    filtered_messages = []
                    user_messages = [msg for msg in messages if msg.get("role") == "user"]
                    
                    # 保留第一个 system 消息
                    for msg in messages:
                        if msg.get("role") == "system":
                            filtered_messages.append(msg)
                            break
                    
                    # 保留最后一个 user 消息（真正的问题）
                    if user_messages:
                        filtered_messages.append(user_messages[-1])
                    
                    if filtered_messages:
                        messages = filtered_messages
                        print(f"  -> Filtered kg task messages: {len(messages)} messages (removed {original_count - len(messages)} demo template messages)")

                # 7.2 通过 memory 机制改写 messages（测试集也使用 memory，但不更新）
                # single_agent
                enhanced_messages = memory_for_enhance.use_memory(task_name, messages)

                # 7.3 通过 execution engine 执行
                history, result = execution_engine.run_sample(
                    task=task_name,
                    index=sample_index,
                    session_id=session_id,
                    messages=enhanced_messages,
                    tools=tools,
                    agent_pool=llm_agent,
                    backend_client=backend,
                )

                # 7.3.1 确保 result 中记录 agent_name（对于 single_agent 也记录）
                if isinstance(result, dict):
                    if isinstance(execution_engine, SingleAgentExecutionEngine):
                        result["agent_name"] = execution_engine.config.agent_name

                # 7.4 测试集：在 offline 模式下不更新记忆（只评估），在 online 模式下更新记忆
                if training_mode == "offline":
                    # offline 模式：测试集不更新记忆（只评估性能）
                    pass
                else:
                    # online 模式：测试集也更新记忆
                    memory.update_memory(task_name, history, result)

                # 7.5 落盘到 test 目录
                # 确保 agent_name 在顶层（从 result 中提取，如果存在）
                agent_name = None
                if isinstance(result, dict):
                    agent_name = result.get("agent_name")
                
                task_dir = ensure_output_dir(test_output_root / task_name)
                out_path = task_dir / f"{sample_index}.json"
                output_data = {
                            "task": task_name,
                            "index": sample_index,
                            "split": "test",
                            "history": history,
                            "result": result,
                }
                if agent_name:
                    output_data["agent_name"] = agent_name
                
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                
                # 记录执行顺序
                if task_name not in execution_order_test:
                    execution_order_test[task_name] = []
                execution_order_test[task_name].append({
                    "task": task_name,
                    "index": sample_index,
                    "split": "test",
                    "execution_order": len(execution_order_test[task_name]) + 1,
                    "timestamp": time.time(),
                    "status": result.get("status", "unknown") if isinstance(result, dict) else "unknown",
                })

                # 7.6 输出 agent 信息
                agent_info = ""
                if isinstance(result, dict) and "agent_name" in result:
                    agent_name = result.get("agent_name", "unknown")
                    agent_info = f", agent={agent_name}"

                print(f"  -> saved to {out_path.relative_to(ROOT_DIR)}{agent_info}\n")

            except Exception as e:
                # 捕获所有异常，记录错误但继续处理下一个样本
                error_msg = f"  -> ERROR: Failed to process sample {sample_index} of task {task_name}: {str(e)}"
                print(error_msg)
                logging.error(error_msg, exc_info=True)
                
                # 可选：保存错误信息到文件
                task_dir = ensure_output_dir(test_output_root / task_name)
                error_path = task_dir / f"{sample_index}.error.json"
                with error_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "task": task_name,
                            "index": sample_index,
                            "split": "test",
                            "error": str(e),
                            "error_type": type(e).__name__,
                        },
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                
                # 记录执行顺序（即使出错也记录）
                if task_name not in execution_order_test:
                    execution_order_test[task_name] = []
                execution_order_test[task_name].append({
                    "task": task_name,
                    "index": sample_index,
                    "split": "test",
                    "execution_order": len(execution_order_test[task_name]) + 1,
                    "timestamp": time.time(),
                    "status": "error",
                    "error": str(e),
                })
                
                print(f"  -> error saved to {error_path.relative_to(ROOT_DIR)}\n")
                continue  # 跳过当前样本，继续下一个

            # 为了更保险，样本之间也稍微停顿一下（完全串行执行）
            time.sleep(1.0)

    # 8) 保存执行顺序文件
    # online模式：在上一级目录（base_output_root）保存 execution_order.json（包含所有任务的执行顺序）
    # offline模式：在train和test目录下分别保存 execution_order.json（包含所有任务的执行顺序）
    # transfer模式：在transfer_train和forward_transfer_test目录下分别保存 execution_order.json
    if execution_order_train:
        # 合并所有任务的执行顺序，按 timestamp 排序
        all_train_orders = []
        for task_name, order_list in execution_order_train.items():
            all_train_orders.extend(order_list)
        # 按 timestamp 排序
        all_train_orders.sort(key=lambda x: x.get("timestamp", 0))
        # 重新分配 execution_order（全局顺序）
        for idx, order_item in enumerate(all_train_orders, start=1):
            order_item["execution_order"] = idx

        if test_schedule:
            # offline模式：在train目录下保存（包含所有任务）
            order_path = train_output_root / "execution_order.json"
            with order_path.open("w", encoding="utf-8") as f:
                json.dump(all_train_orders, f, indent=2, ensure_ascii=False)
            print(f"[Execution Order] Saved train execution order: {len(all_train_orders)} samples from {len(execution_order_train)} task(s) -> {order_path.relative_to(ROOT_DIR)}")
        else:
            # online/transfer模式：在当前目录保存
            if training_mode == "transfer":
                # transfer 前向迁移模式：train_output_root 已经是 transfer_train 目录
                order_path = train_output_root / "execution_order.json"
            else:
                # online模式：在上一级目录（base_output_root）保存
                order_path = base_output_root / "execution_order.json"
            with order_path.open("w", encoding="utf-8") as f:
                json.dump(all_train_orders, f, indent=2, ensure_ascii=False)
            print(f"[Execution Order] Saved execution order: {len(all_train_orders)} samples from {len(execution_order_train)} task(s) -> {order_path.relative_to(ROOT_DIR)}")
    
    if execution_order_test:
        # offline模式：在test目录下保存（包含所有任务）
        # replay模式：立即测试保存到test/目录，replay测试保存到各自的replayX/test/目录
        # 合并所有任务的执行顺序，按 timestamp 排序
        all_test_orders = []
        for task_name, order_list in execution_order_test.items():
            all_test_orders.extend(order_list)
        # 按 timestamp 排序
        all_test_orders.sort(key=lambda x: x.get("timestamp", 0))
        # 重新分配 execution_order（全局顺序）
        for idx, order_item in enumerate(all_test_orders, start=1):
            order_item["execution_order"] = idx

        # replay模式：需要分别保存立即测试和replay测试的execution_order
        if training_mode == "replay":
            # 分离立即测试和replay测试
            immediate_test_orders = [o for o in all_test_orders if o.get("split") == "immediate_test"]
            replay_test_orders = [o for o in all_test_orders if o.get("split") == "test" and "replay_id" in o]

            # 保存立即测试的execution_order到test/目录
            if immediate_test_orders:
                # 重新分配execution_order
                for idx, order_item in enumerate(immediate_test_orders, start=1):
                    order_item["execution_order"] = idx
                immediate_test_order_path = train_output_root / "test" / "execution_order.json"
                ensure_output_dir(train_output_root / "test")
                with immediate_test_order_path.open("w", encoding="utf-8") as f:
                    json.dump(immediate_test_orders, f, indent=2, ensure_ascii=False)
                print(f"[Execution Order] Saved immediate test execution order: {len(immediate_test_orders)} samples -> {immediate_test_order_path.relative_to(ROOT_DIR)}")

            # 保存replay测试的execution_order到各自的replayX/test/目录
            if replay_test_orders:
                # 按replay_id分组
                replay_groups = {}
                for order_item in replay_test_orders:
                    replay_id = order_item.get("replay_id")
                    if replay_id not in replay_groups:
                        replay_groups[replay_id] = []
                    replay_groups[replay_id].append(order_item)

                # 为每个replay保存execution_order
                for replay_id, orders in replay_groups.items():
                    # 重新分配execution_order
                    for idx, order_item in enumerate(orders, start=1):
                        order_item["execution_order"] = idx
                    replay_test_order_path = train_output_root / f"replay{replay_id}" / "test" / "execution_order.json"
                    ensure_output_dir(train_output_root / f"replay{replay_id}" / "test")
                    with replay_test_order_path.open("w", encoding="utf-8") as f:
                        json.dump(orders, f, indent=2, ensure_ascii=False)
                    print(f"[Execution Order] Saved replay{replay_id} test execution order: {len(orders)} samples -> {replay_test_order_path.relative_to(ROOT_DIR)}")
        else:
            # 非replay模式：使用原有逻辑
            # 如果 test_output_root 为 None，使用 train_output_root
            if test_output_root is not None:
                order_path = test_output_root / "execution_order.json"
            else:
                order_path = train_output_root / "execution_order_test.json"
            with order_path.open("w", encoding="utf-8") as f:
                json.dump(all_test_orders, f, indent=2, ensure_ascii=False)
            print(f"[Execution Order] Saved test execution order: {len(all_test_orders)} samples from {len(execution_order_test)} task(s) -> {order_path.relative_to(ROOT_DIR)}")


    # Transfer 模式：保存 forward_transfer_test 的执行顺序
    if execution_order_forward_test:
        # 合并所有任务的执行顺序，按 timestamp 排序
        all_forward_test_orders = []
        for task_name, order_list in execution_order_forward_test.items():
            all_forward_test_orders.extend(order_list)
        # 按 timestamp 排序
        all_forward_test_orders.sort(key=lambda x: x.get("timestamp", 0))
        # 重新分配 execution_order（全局顺序）
        for idx, order_item in enumerate(all_forward_test_orders, start=1):
            order_item["execution_order"] = idx

        # 保存到 base_output_root / forward_transfer_test 目录（与文件保存位置一致）
        forward_test_order_path = base_output_root / "forward_transfer_test" / "execution_order.json"
        ensure_output_dir(base_output_root / "forward_transfer_test")
        with forward_test_order_path.open("w", encoding="utf-8") as f:
            json.dump(all_forward_test_orders, f, indent=2, ensure_ascii=False)
        print(f"[Execution Order] Saved forward transfer test execution order: {len(all_forward_test_orders)} samples from {len(execution_order_forward_test)} task(s) -> {forward_test_order_path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Optional

import yaml

from .MEMs import MEMs, MEMsConfig, MemorySourceConfig


def load_mems_from_yaml(yaml_path: str) -> Optional[MEMs]:
    """从YAML配置文件加载MEMs实例"""
    config_path = Path(yaml_path)
    if not config_path.exists():
        raise FileNotFoundError(f"MEMs config not found: {yaml_path}")

    with config_path.open("r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f) or {}

    mems_cfg = yaml_data.get("mems", {})
    if not mems_cfg:
        raise ValueError(f"'mems' section not found in {yaml_path}")

    # 解析记忆源配置
    source_1_cfg = mems_cfg.get("memory_source_1", {})
    source_2_cfg = mems_cfg.get("memory_source_2", {})

    if not source_1_cfg or not source_2_cfg:
        raise ValueError("Both memory_source_1 and memory_source_2 must be configured")

    # 构建配置对象
    config = MEMsConfig(
        model_name=mems_cfg.get("model_name", ""),
        memory_source_1=MemorySourceConfig(
            name=source_1_cfg.get("name", "memory_1"),
            config_path=Path(source_1_cfg.get("config_path", ""))
        ),
        memory_source_2=MemorySourceConfig(
            name=source_2_cfg.get("name", "memory_2"),
            config_path=Path(source_2_cfg.get("config_path", ""))
        ),
        retrieval_trigger_prompt=mems_cfg.get("retrieval_trigger_prompt", ""),
        update_trigger_prompt=mems_cfg.get("update_trigger_prompt", ""),
        trigger_model_max_retries=mems_cfg.get("trigger_model_max_retries", 5),
        update_success_only=mems_cfg.get("update_success_only", True),
        update_reward_bigger_than_zero=mems_cfg.get("update_reward_bigger_than_zero", True),
    )

    return MEMs(config)


__all__ = ["MEMs", "MEMsConfig", "MemorySourceConfig", "load_mems_from_yaml"]

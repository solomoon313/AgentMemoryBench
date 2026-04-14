#!/usr/bin/env python3
"""
计算累积成功率脚本
根据 execution_order 计算累积成功率，适配 system memory 和 personal memory 任务
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 尝试导入 matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plotting will be disabled")


def is_success_for_personal_memory(result_data: Dict) -> bool:
    """
    判断 personal memory 任务是否成功
    Personal memory 任务（如 locomo）的成功判断：
    优先使用 llm_score > 0.5 来判断成功
    - result.metrics.llm_score > 0.5 或
    - history 中最后一项的 metrics.llm_score > 0.5
    """
    # 优先检查 result.metrics.llm_score
    if "result" in result_data:
        result = result_data["result"]
        if "metrics" in result:
            metrics = result["metrics"]
            if "llm_score" in metrics:
                llm_score = metrics["llm_score"]
                if isinstance(llm_score, (int, float)):
                    return llm_score > 0.5
    
    # 如果 result.metrics.llm_score 不存在，检查 history 中最后一项的 metrics.llm_score
    if "history" in result_data and len(result_data["history"]) > 0:
        last_item = result_data["history"][-1]
        if "metrics" in last_item:
            metrics = last_item["metrics"]
            if "llm_score" in metrics:
                llm_score = metrics["llm_score"]
                if isinstance(llm_score, (int, float)):
                    return llm_score > 0.5
    
    # 如果都没有 llm_score，返回 False
    return False


def is_success_for_system_memory(result_data: Dict) -> bool:
    """
    判断 system memory 任务是否成功
    System memory 任务（如 db, os, kg, webshop, alfworld）的成功判断：
    - result.reward == 1 或
    - result.status == "completed" 且 result.reward == 1
    """
    if "result" in result_data:
        result = result_data["result"]
        if "reward" in result:
            reward = result["reward"]
            if isinstance(reward, (int, float)) and reward == 1:
                return True
        # 如果没有 reward，检查 status
        if "status" in result:
            status = result["status"]
            if status == "completed" and result.get("reward", 0) == 1:
                return True
    
    return False


def is_personal_memory_task(task_name: str) -> bool:
    """判断任务是否是 personal memory 任务"""
    return "locomo" in task_name.lower()


def is_system_memory_task(task_name: str) -> bool:
    """判断任务是否是 system memory 任务"""
    task_lower = task_name.lower()
    return any(x in task_lower for x in ["db", "os", "kg", "webshop", "alfworld", "dbbench", "osbench", "kgbench"])


def has_subdirectory_structure(output_dir: Path, execution_order: List[Dict]) -> bool:
    """
    检查输出目录是否使用子目录结构（每个任务一个子目录）
    """
    if not execution_order:
        return False

    # 取第一个样本，检查文件是否在子目录中
    first_item = execution_order[0]
    task_name = first_item.get("task", "")
    index = first_item.get("index")

    if not task_name or index is None:
        return False

    # 检查子目录中的文件是否存在
    task_dir = output_dir / task_name
    result_file_in_subdir = task_dir / f"{index}.json"

    # 检查主目录中的文件是否存在
    result_file_in_main = output_dir / f"{index}.json"

    # 如果子目录中的文件存在，说明使用子目录结构
    if result_file_in_subdir.exists():
        return True
    # 如果主目录中的文件存在，说明不使用子目录结构
    elif result_file_in_main.exists():
        return False

    # 如果都不存在，默认返回 False
    return False


def detect_task_type(output_dir: Path) -> str:
    """
    根据任务名称或数据特征检测任务类型
    返回 "personal"、"system" 或 "mixed"
    """
    # 优先读取 execution_order.json 来准确检测混合场景
    execution_order_file = output_dir / "execution_order.json"
    if execution_order_file.exists():
        with open(execution_order_file, 'r', encoding='utf-8') as f:
            execution_order = json.load(f)
            if execution_order and len(execution_order) > 0:
                # 收集所有任务名称
                task_names = set()
                for item in execution_order:
                    task = item.get("task", "")
                    if task:
                        task_names.add(task)

                # 检查是否同时包含 system 和 personal memory 任务
                has_system = any(is_system_memory_task(t) for t in task_names)
                has_personal = any(is_personal_memory_task(t) for t in task_names)

                if has_system and has_personal:
                    return "mixed"
                elif has_personal:
                    return "personal"
                elif has_system:
                    return "system"

    # 如果无法从 execution_order.json 判断，尝试从路径中提取任务名称
    parts = output_dir.parts
    has_system_in_path = False
    has_personal_in_path = False

    for part in parts:
        if part.startswith("locomo") or "locomo" in part.lower():
            has_personal_in_path = True
        if part in ["db", "os", "kg", "webshop", "alfworld"] or "dbbench" in part or "osbench" in part or "kgbench" in part:
            has_system_in_path = True

    if has_system_in_path and has_personal_in_path:
        return "mixed"
    elif has_personal_in_path:
        return "personal"
    elif has_system_in_path:
        return "system"

    # 默认返回 system（更常见）
    return "system"


def load_execution_order(output_dir: Path) -> List[Dict]:
    """加载 execution_order.json"""
    execution_order_file = output_dir / "execution_order.json"
    if not execution_order_file.exists():
        raise FileNotFoundError(f"execution_order.json not found in {output_dir}")
    
    with open(execution_order_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_sample_result(output_dir: Path, index: int, task_name: Optional[str] = None) -> Optional[Dict]:
    """
    加载单个样本的结果文件
    如果是混合场景，需要从对应的任务子目录加载
    """
    # 如果是混合场景，需要从任务子目录加载
    if task_name:
        task_dir = output_dir / task_name
        result_file = task_dir / f"{index}.json"
    else:
        result_file = output_dir / f"{index}.json"
    
    if not result_file.exists():
        return None
    
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load {result_file}: {e}")
        return None


def calculate_cumulative_success_rate_single(
    execution_order: List[Dict],
    output_dir: Path,
    is_success_func,
    task_filter: Optional[str] = None,
    category_filter: Optional[int] = None,
    is_mixed: bool = False
) -> Tuple[List[float], List[int], List[int], List[Dict]]:
    """
    计算单个任务或过滤后任务的累积成功率
    
    Args:
        execution_order: 执行顺序列表
        output_dir: 输出目录路径
        is_success_func: 成功判断函数
        task_filter: 任务名称过滤（如果指定，只计算该任务的样本）
        category_filter: category 过滤（如果指定，只计算该 category 的样本）
        is_mixed: 是否是混合场景（需要从任务子目录加载文件）
    
    Returns:
        (cumulative_rates, success_counts, total_counts, results_data)
    """
    cumulative_rates = []
    success_counts = []
    total_counts = []
    results_data = []
    
    cumulative_success = 0
    cumulative_total = 0
    
    for order_item in execution_order:
        # 如果指定了任务过滤，跳过不匹配的任务
        if task_filter is not None:
            task_name = order_item.get("task", "")
            if task_name != task_filter:
                continue
        
        index = order_item.get("index")
        if index is None:
            continue
        
        # 加载样本结果（混合场景需要从任务子目录加载）
        task_name = order_item.get("task", "") if is_mixed else None
        result_data = load_sample_result(output_dir, index, task_name)
        if result_data is None:
            continue
        
        # 如果指定了 category 过滤，检查 category 是否匹配
        if category_filter is not None:
            category = result_data.get("result", {}).get("category")
            if category != category_filter:
                continue
        
        # 判断是否成功
        is_success = is_success_func(result_data)
        
        cumulative_total += 1
        if is_success:
            cumulative_success += 1
        
        cumulative_rate = cumulative_success / cumulative_total if cumulative_total > 0 else 0.0
        
        cumulative_rates.append(cumulative_rate)
        success_counts.append(cumulative_success)
        total_counts.append(cumulative_total)
        
        results_data.append({
            "execution_order": order_item.get("execution_order"),
            "index": index,
            "task": order_item.get("task"),
            "split": order_item.get("split"),
            "category": result_data.get("result", {}).get("category"),
            "success": is_success,
            "cumulative_success": cumulative_success,
            "cumulative_total": cumulative_total,
            "cumulative_rate": cumulative_rate
        })
    
    return cumulative_rates, success_counts, total_counts, results_data


def calculate_cumulative_success_rate_mixed(
    execution_order: List[Dict],
    output_dir: Path,
    output_file: Optional[Path],
    plot_file: Optional[Path]
) -> Dict:
    """
    计算混合场景（system memory + personal memory）的累积成功率
    
    Args:
        execution_order: 执行顺序列表
        output_dir: 输出目录路径
        output_file: 输出文件路径（可选）
        plot_file: 绘图文件路径（可选）
    
    Returns:
        结果字典
    """
    # 分离 system memory 和 personal memory 任务
    system_tasks = []
    personal_tasks = []
    
    for item in execution_order:
        task_name = item.get("task", "")
        if is_system_memory_task(task_name):
            system_tasks.append(item)
        elif is_personal_memory_task(task_name):
            personal_tasks.append(item)
    
    print(f"Mixed scenario detected:")
    print(f"  System memory tasks: {len(system_tasks)} samples")
    print(f"  Personal memory tasks: {len(personal_tasks)} samples")
    
    # 计算 system memory 任务的累积成功率
    system_result = None
    if system_tasks:
        cumulative_rates_sys, success_counts_sys, total_counts_sys, results_data_sys = calculate_cumulative_success_rate_single(
            system_tasks, output_dir, is_success_for_system_memory, is_mixed=True
        )
        
        cumulative_rate_sys = cumulative_rates_sys[-1] if cumulative_rates_sys else 0.0
        cumulative_success_sys = success_counts_sys[-1] if success_counts_sys else 0
        cumulative_total_sys = total_counts_sys[-1] if total_counts_sys else 0
        
        system_result = {
            "total_samples": cumulative_total_sys,
            "total_success": cumulative_success_sys,
            "final_success_rate": cumulative_rate_sys,
            "cumulative_rates": cumulative_rates_sys,
            "success_counts": success_counts_sys,
            "total_counts": total_counts_sys,
            "detailed_results": results_data_sys
        }
        print(f"  System memory: {cumulative_total_sys} samples, {cumulative_success_sys} success, {cumulative_rate_sys:.4f} ({cumulative_rate_sys*100:.2f}%)")
    
    # 计算 personal memory 任务的累积成功率（按 category）
    personal_result = None
    personal_sub_tasks = {}
    
    if personal_tasks:
        # 先读取所有 personal memory 样本的 category
        categories = set()
        for order_item in personal_tasks:
            index = order_item.get("index")
            task_name = order_item.get("task", "")
            if index is None:
                continue
            result_data = load_sample_result(output_dir, index, task_name)
            if result_data is None:
                continue
            category = result_data.get("result", {}).get("category")
            if category is not None:
                categories.add(category)
        
        categories = sorted(categories)
        print(f"  Personal memory categories: {categories}")
        
        # 计算每个 category 的累积成功率
        for category in categories:
            cumulative_rates_cat, success_counts_cat, total_counts_cat, results_data_cat = calculate_cumulative_success_rate_single(
                personal_tasks, output_dir, is_success_for_personal_memory, 
                category_filter=category, is_mixed=True
            )
            
            cumulative_rate_cat = cumulative_rates_cat[-1] if cumulative_rates_cat else 0.0
            cumulative_success_cat = success_counts_cat[-1] if success_counts_cat else 0
            cumulative_total_cat = total_counts_cat[-1] if total_counts_cat else 0
            
            category_name = f"Category {category}"
            personal_sub_tasks[category_name] = {
                "category": category,
                "task_name": category_name,
                "total_samples": cumulative_total_cat,
                "total_success": cumulative_success_cat,
                "final_success_rate": cumulative_rate_cat,
                "cumulative_rates": cumulative_rates_cat,
                "success_counts": success_counts_cat,
                "total_counts": total_counts_cat,
                "detailed_results": results_data_cat
            }
            print(f"    Category {category}: {cumulative_total_cat} samples, {cumulative_success_cat} success, {cumulative_rate_cat:.4f} ({cumulative_rate_cat*100:.2f}%)")
        
        # 计算 personal memory 总体的累积成功率
        cumulative_rates_pers, success_counts_pers, total_counts_pers, results_data_pers = calculate_cumulative_success_rate_single(
            personal_tasks, output_dir, is_success_for_personal_memory, is_mixed=True
        )
        
        cumulative_rate_pers = cumulative_rates_pers[-1] if cumulative_rates_pers else 0.0
        cumulative_success_pers = success_counts_pers[-1] if success_counts_pers else 0
        cumulative_total_pers = total_counts_pers[-1] if total_counts_pers else 0
        
        personal_result = {
            "sub_tasks": personal_sub_tasks,
            "overall": {
                "total_samples": cumulative_total_pers,
                "total_success": cumulative_success_pers,
                "final_success_rate": cumulative_rate_pers,
                "cumulative_rates": cumulative_rates_pers,
                "success_counts": success_counts_pers,
                "total_counts": total_counts_pers,
                "detailed_results": results_data_pers
            }
        }
        print(f"  Personal memory overall: {cumulative_total_pers} samples, {cumulative_success_pers} success, {cumulative_rate_pers:.4f} ({cumulative_rate_pers*100:.2f}%)")
    
    # 计算总体的累积成功率（所有任务混合）
    # 需要统一判断函数：根据任务类型选择不同的判断函数
    def mixed_is_success(result_data: Dict, task_name: str) -> bool:
        if is_system_memory_task(task_name):
            return is_success_for_system_memory(result_data)
        elif is_personal_memory_task(task_name):
            return is_success_for_personal_memory(result_data)
        return False
    
    # 为每个任务项添加判断函数
    cumulative_rates_overall = []
    success_counts_overall = []
    total_counts_overall = []
    results_data_overall = []
    
    cumulative_success = 0
    cumulative_total = 0
    
    for order_item in execution_order:
        task_name = order_item.get("task", "")
        index = order_item.get("index")
        if index is None:
            continue
        
        result_data = load_sample_result(output_dir, index, task_name)
        if result_data is None:
            continue
        
        is_success = mixed_is_success(result_data, task_name)
        
        cumulative_total += 1
        if is_success:
            cumulative_success += 1
        
        cumulative_rate = cumulative_success / cumulative_total if cumulative_total > 0 else 0.0
        
        cumulative_rates_overall.append(cumulative_rate)
        success_counts_overall.append(cumulative_success)
        total_counts_overall.append(cumulative_total)
        
        results_data_overall.append({
            "execution_order": order_item.get("execution_order"),
            "index": index,
            "task": task_name,
            "split": order_item.get("split"),
            "category": result_data.get("result", {}).get("category"),
            "success": is_success,
            "cumulative_success": cumulative_success,
            "cumulative_total": cumulative_total,
            "cumulative_rate": cumulative_rate
        })
    
    cumulative_rate_overall = cumulative_rates_overall[-1] if cumulative_rates_overall else 0.0
    cumulative_success_overall = success_counts_overall[-1] if success_counts_overall else 0
    cumulative_total_overall = total_counts_overall[-1] if total_counts_overall else 0
    
    overall_result = {
        "total_samples": cumulative_total_overall,
        "total_success": cumulative_success_overall,
        "final_success_rate": cumulative_rate_overall,
        "cumulative_rates": cumulative_rates_overall,
        "success_counts": success_counts_overall,
        "total_counts": total_counts_overall,
        "detailed_results": results_data_overall
    }
    
    print(f"  Overall (mixed): {cumulative_total_overall} samples, {cumulative_success_overall} success, {cumulative_rate_overall:.4f} ({cumulative_rate_overall*100:.2f}%)")
    
    result = {
        "task_type": "mixed",
        "output_dir": str(output_dir),
        "system_memory": system_result,
        "personal_memory": personal_result,
        "overall": overall_result
    }
    
    # 绘图
    if plot_file:
        plot_cumulative_success_rate_mixed(result, plot_file)
    
    # 输出结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Task Type: mixed")
    if system_result:
        print(f"System Memory - Total Samples: {system_result['total_samples']}")
        print(f"System Memory - Total Success: {system_result['total_success']}")
        print(f"System Memory - Final Success Rate: {system_result['final_success_rate']:.4f} ({system_result['final_success_rate']*100:.2f}%)")
    if personal_result:
        print(f"Personal Memory - Total Samples: {personal_result['overall']['total_samples']}")
        print(f"Personal Memory - Total Success: {personal_result['overall']['total_success']}")
        print(f"Personal Memory - Final Success Rate: {personal_result['overall']['final_success_rate']:.4f} ({personal_result['overall']['final_success_rate']*100:.2f}%)")
    print(f"Overall (Mixed) - Total Samples: {overall_result['total_samples']}")
    print(f"Overall (Mixed) - Total Success: {overall_result['total_success']}")
    print(f"Overall (Mixed) - Final Success Rate: {overall_result['final_success_rate']:.4f} ({overall_result['final_success_rate']*100:.2f}%)")
    print(f"{'='*60}\n")
    
    return result


def calculate_cumulative_success_rate(
    output_dir: Path,
    task_type: Optional[str] = None,
    output_file: Optional[Path] = None,
    plot_file: Optional[Path] = None
) -> Dict:
    """
    计算累积成功率
    
    Args:
        output_dir: 输出目录路径
        task_type: 任务类型 ("personal"、"system" 或 "mixed")，如果为 None 则自动检测
        output_file: 输出文件路径（可选）
        plot_file: 绘图文件路径（可选）
    
    Returns:
        结果字典
    """
    # 检测任务类型
    if task_type is None:
        task_type = detect_task_type(output_dir)
        print(f"Detected task type: {task_type}")
    
    # 加载执行顺序
    execution_order = load_execution_order(output_dir)
    
    # 按 execution_order 排序
    execution_order.sort(key=lambda x: x.get("execution_order", 0))

    # 检查是否使用子目录结构
    use_subdirs = has_subdirectory_structure(output_dir, execution_order)
    if use_subdirs:
        print(f"Detected subdirectory structure: using task subdirectories for data files")

    if task_type == "mixed":
        # 混合场景：分别计算 system memory 和 personal memory 任务
        return calculate_cumulative_success_rate_mixed(
            execution_order, output_dir, output_file, plot_file
        )

    # 选择成功判断函数
    is_success_func = is_success_for_personal_memory if task_type == "personal" else is_success_for_system_memory

    if task_type == "system":
        # System memory 任务：直接计算总体
        cumulative_rates, success_counts, total_counts, results_data = calculate_cumulative_success_rate_single(
            execution_order, output_dir, is_success_func, is_mixed=use_subdirs
        )
        
        cumulative_rate = cumulative_rates[-1] if cumulative_rates else 0.0
        cumulative_success = success_counts[-1] if success_counts else 0
        cumulative_total = total_counts[-1] if total_counts else 0
        
        result = {
            "task_type": task_type,
            "output_dir": str(output_dir),
            "total_samples": cumulative_total,
            "total_success": cumulative_success,
            "final_success_rate": cumulative_rate,
            "cumulative_rates": cumulative_rates,
            "success_counts": success_counts,
            "total_counts": total_counts,
            "detailed_results": results_data
        }
        
        # 绘图
        if plot_file:
            plot_cumulative_success_rate_system(result, plot_file)
        
    else:  # task_type == "personal"
        # Personal memory 任务：按 category 分组计算
        # 先读取所有样本的 category，提取所有 category 值
        categories = set()
        for order_item in execution_order:
            index = order_item.get("index")
            if index is None:
                continue
            task_name = order_item.get("task", "") if use_subdirs else None
            result_data = load_sample_result(output_dir, index, task_name)
            if result_data is None:
                continue
            category = result_data.get("result", {}).get("category")
            if category is not None:
                categories.add(category)

        categories = sorted(categories)
        print(f"Found {len(categories)} categories: {categories}")

        # 计算每个 category 的累积成功率
        sub_task_results = {}
        for category in categories:
            cumulative_rates, success_counts, total_counts, results_data = calculate_cumulative_success_rate_single(
                execution_order, output_dir, is_success_func, category_filter=category, is_mixed=use_subdirs
            )
            
            cumulative_rate = cumulative_rates[-1] if cumulative_rates else 0.0
            cumulative_success = success_counts[-1] if success_counts else 0
            cumulative_total = total_counts[-1] if total_counts else 0
            
            category_name = f"Category {category}"
            sub_task_results[category_name] = {
                "category": category,
                "task_name": category_name,
                "total_samples": cumulative_total,
                "total_success": cumulative_success,
                "final_success_rate": cumulative_rate,
                "cumulative_rates": cumulative_rates,
                "success_counts": success_counts,
                "total_counts": total_counts,
                "detailed_results": results_data
            }
            
            print(f"  Category {category}: {cumulative_total} samples, {cumulative_success} success, {cumulative_rate:.4f} ({cumulative_rate*100:.2f}%)")
        
        # 计算总体的累积成功率
        cumulative_rates_overall, success_counts_overall, total_counts_overall, results_data_overall = calculate_cumulative_success_rate_single(
            execution_order, output_dir, is_success_func, is_mixed=use_subdirs
        )
        
        cumulative_rate_overall = cumulative_rates_overall[-1] if cumulative_rates_overall else 0.0
        cumulative_success_overall = success_counts_overall[-1] if success_counts_overall else 0
        cumulative_total_overall = total_counts_overall[-1] if total_counts_overall else 0
        
        result = {
            "task_type": task_type,
            "output_dir": str(output_dir),
            "sub_tasks": sub_task_results,
            "overall": {
                "total_samples": cumulative_total_overall,
                "total_success": cumulative_success_overall,
                "final_success_rate": cumulative_rate_overall,
                "cumulative_rates": cumulative_rates_overall,
                "success_counts": success_counts_overall,
                "total_counts": total_counts_overall,
                "detailed_results": results_data_overall
            }
        }
        
        # 绘图
        if plot_file:
            plot_cumulative_success_rate_personal(result, plot_file)
    
    # 输出结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print(f"Task Type: {task_type}")
    if task_type == "system":
        print(f"Total Samples: {result['total_samples']}")
        print(f"Total Success: {result['total_success']}")
        print(f"Final Success Rate: {result['final_success_rate']:.4f} ({result['final_success_rate']*100:.2f}%)")
    else:
        print(f"Overall - Total Samples: {result['overall']['total_samples']}")
        print(f"Overall - Total Success: {result['overall']['total_success']}")
        print(f"Overall - Final Success Rate: {result['overall']['final_success_rate']:.4f} ({result['overall']['final_success_rate']*100:.2f}%)")
    print(f"{'='*60}\n")
    
    return result


def plot_cumulative_success_rate_system(result: Dict, plot_file: Path):
    """绘制 system memory 任务的累积成功率图"""
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    cumulative_rates = result["cumulative_rates"]
    total_counts = result["total_counts"]
    
    plt.figure(figsize=(10, 6))
    plt.plot(total_counts, cumulative_rates, linewidth=2, color='#2E86AB')
    plt.xlabel('Number of Samples', fontsize=12)
    plt.ylabel('Cumulative Success Rate', fontsize=12)
    plt.title(f'Cumulative Success Rate (System Memory Task)\nFinal Rate: {result["final_success_rate"]:.2%}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.tight_layout()
    
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_file}")


def plot_cumulative_success_rate_mixed(result: Dict, plot_file: Path):
    """
    绘制混合场景（system memory + personal memory）的累积成功率图
    布局说明：
    - 第一行：System memory（如果有）+ Personal memory categories（如果有）
    - 第二行：Personal memory overall（如果有）+ Overall mixed + 空白
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    system_result = result.get("system_memory")
    personal_result = result.get("personal_memory")
    overall_result = result.get("overall")
    
    # 计算需要绘制的子图数量
    num_plots = 0
    if system_result:
        num_plots += 1
    if personal_result:
        sub_tasks = personal_result.get("sub_tasks", {})
        num_plots += len(sub_tasks)  # 每个 category 一个图
        num_plots += 1  # personal memory overall
    num_plots += 1  # overall mixed
    
    # 确定子图布局（尽量使用 2x3 或 3x3）
    if num_plots <= 6:
        rows, cols = 2, 3
    else:
        rows, cols = 3, 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    
    # 1. System memory 任务
    if system_result:
        ax = axes[plot_idx]
        cumulative_rates = system_result["cumulative_rates"]
        total_counts = system_result["total_counts"]
        
        ax.plot(total_counts, cumulative_rates, linewidth=2, color='#2E86AB')
        ax.set_xlabel('Number of Samples', fontsize=10)
        ax.set_ylabel('Cumulative Success Rate', fontsize=10)
        ax.set_title(f'System Memory Tasks\nFinal Rate: {system_result["final_success_rate"]:.2%}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plot_idx += 1
    
    # 2. Personal memory 任务（按 category）
    if personal_result:
        sub_tasks = personal_result.get("sub_tasks", {})
        sorted_category_names = sorted(sub_tasks.keys(), key=lambda x: sub_tasks[x].get("category", 0))
        
        for category_name in sorted_category_names:
            if plot_idx >= len(axes):
                break
            category_result = sub_tasks[category_name]
            ax = axes[plot_idx]
            cumulative_rates = category_result["cumulative_rates"]
            total_counts = category_result["total_counts"]
            
            category_num = category_result.get("category", 0)
            ax.plot(total_counts, cumulative_rates, linewidth=2, color=f'C{plot_idx}')
            ax.set_xlabel('Number of Samples', fontsize=10)
            ax.set_ylabel('Cumulative Success Rate', fontsize=10)
            ax.set_title(f'Personal Memory - Category {category_num}\nFinal Rate: {category_result["final_success_rate"]:.2%}', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            plot_idx += 1
        
        # Personal memory overall
        if plot_idx < len(axes):
            ax = axes[plot_idx]
            overall_pers = personal_result.get("overall", {})
            cumulative_rates = overall_pers.get("cumulative_rates", [])
            total_counts = overall_pers.get("total_counts", [])
            
            ax.plot(total_counts, cumulative_rates, linewidth=2, color='#E63946')
            ax.set_xlabel('Number of Samples', fontsize=10)
            ax.set_ylabel('Cumulative Success Rate', fontsize=10)
            ax.set_title(f'Personal Memory - Overall\nFinal Rate: {overall_pers.get("final_success_rate", 0):.2%}', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            plot_idx += 1
    
    # 3. Overall mixed
    if plot_idx < len(axes) and overall_result:
        ax = axes[plot_idx]
        cumulative_rates = overall_result["cumulative_rates"]
        total_counts = overall_result["total_counts"]
        
        ax.plot(total_counts, cumulative_rates, linewidth=2, color='#F77F00')
        ax.set_xlabel('Number of Samples', fontsize=10)
        ax.set_ylabel('Cumulative Success Rate', fontsize=10)
        ax.set_title(f'Overall (Mixed: System + Personal)\nFinal Rate: {overall_result["final_success_rate"]:.2%}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plot_idx += 1
    
    # 隐藏未使用的子图
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    # 添加总标题
    title_parts = []
    if system_result:
        title_parts.append("System Memory")
    if personal_result:
        title_parts.append("Personal Memory")
    title_parts.append("Mixed Overall")
    
    plt.suptitle('Cumulative Success Rate (Mixed Scenario)\n' + 
                 ' | '.join(title_parts), 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_file}")
    print(f"  Total plots: {plot_idx}")


def plot_cumulative_success_rate_personal(result: Dict, plot_file: Path):
    """
    绘制 personal memory 任务的累积成功率图（五个子图）
    布局说明：
    - 位置 0-3：四个 category（Category 1, Category 2, Category 3, Category 4）
    - 位置 4：总体累积成功率（Overall）
    - 位置 5：隐藏（未使用）
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping plot")
        return
    
    sub_tasks = result["sub_tasks"]
    overall = result["overall"]
    
    # 创建 2x3 的子图布局（5个图：4个 category + 1个总体）
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 按 category 排序，确保顺序一致（Category 1, Category 2, Category 3, Category 4）
    sorted_category_names = sorted(sub_tasks.keys(), key=lambda x: sub_tasks[x].get("category", 0))
    
    # 绘制每个 category 的图
    for idx, category_name in enumerate(sorted_category_names):
        category_result = sub_tasks[category_name]
        ax = axes[idx]
        cumulative_rates = category_result["cumulative_rates"]
        total_counts = category_result["total_counts"]
        
        ax.plot(total_counts, cumulative_rates, linewidth=2, color=f'C{idx}')
        ax.set_xlabel('Number of Samples', fontsize=10)
        ax.set_ylabel('Cumulative Success Rate', fontsize=10)
        # 明确标注 category 和最终成功率
        category_num = category_result.get("category", idx + 1)
        ax.set_title(f'Category {category_num} (Sub-task {idx+1}/{len(sorted_category_names)})\nFinal Rate: {category_result["final_success_rate"]:.2%}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    # 绘制总体图（位置4）
    ax = axes[4]
    cumulative_rates_overall = overall["cumulative_rates"]
    total_counts_overall = overall["total_counts"]
    
    ax.plot(total_counts_overall, cumulative_rates_overall, linewidth=2, color='#E63946')
    ax.set_xlabel('Number of Samples', fontsize=10)
    ax.set_ylabel('Cumulative Success Rate', fontsize=10)
    ax.set_title(f'Overall (All Categories Combined)\nFinal Rate: {overall["final_success_rate"]:.2%}', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # 隐藏第6个子图（位置5）
    axes[5].axis('off')
    
    # 添加总标题，说明图表布局
    category_nums = [str(sub_tasks[name].get("category", "")) for name in sorted_category_names]
    plt.suptitle('Cumulative Success Rate (Personal Memory Tasks)\n'
                 f'Categories: {", ".join(category_nums)} | Overall: All categories combined', 
                 fontsize=16, y=0.995)
    plt.tight_layout()
    
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {plot_file}")
    print(f"  Layout: {len(sorted_category_names)} categories + 1 overall = {len(sorted_category_names)+1} plots")
    print(f"  Categories: {', '.join(category_nums)}")


def main():
    parser = argparse.ArgumentParser(description="计算累积成功率")
    parser.add_argument(
        "output_dir",
        type=str,
        help="输出目录路径（包含 execution_order.json 和样本 JSON 文件）"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["personal", "system", "mixed"],
        default=None,
        help="任务类型：'personal'、'system' 或 'mixed'（如果不指定则自动检测）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件路径（JSON格式，可选，默认：output_dir/cumulative_success.json）"
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="绘图文件路径（PNG格式，可选，默认：output_dir/cumulative_success.png）"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="禁用绘图功能"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Directory {output_dir} does not exist")
        return
    
    # 自动生成输出文件路径（如果未指定）
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = output_dir / "cumulative_success.json"
    
    # 自动生成绘图文件路径（如果未指定且未禁用）
    if args.no_plot:
        plot_file = None
    elif args.plot:
        plot_file = Path(args.plot)
    else:
        plot_file = output_dir / "cumulative_success.png"
    
    try:
        calculate_cumulative_success_rate(
            output_dir=output_dir,
            task_type=args.task_type,
            output_file=output_file,
            plot_file=plot_file
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


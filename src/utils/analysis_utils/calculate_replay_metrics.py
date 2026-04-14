#!/usr/bin/env python3
"""
Calculate two core metrics for Replay mode:
1. Average Success Rate: average performance across all replay stage tests
2. Forgetting Gain: quantifies performance drop from post-learning to subsequent replays

Usage:
    python calculate_replay_metrics.py <replay_dir>

Example:
    python calculate_replay_metrics.py outputs/replay/mem0/seed66-1
"""

import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple
from collections import defaultdict


def load_sample_result(file_path: Path) -> Dict:
    """Load the result file for a single sample."""
    if not file_path.exists():
        return None

    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_performance(result: Dict, metric_type: str = "llm_score") -> float:
    """
    Extract performance metric from the result.

    Args:
        result: sample result dict
        metric_type: metric type, one of "llm_score", "f1_score", "bleu_score"

    Returns:
        performance score (0-1)
    """
    if result is None:
        return 0.0

    metrics = result.get("result", {}).get("metrics", {})

    # Try to get reward from history (for train immediate test)
    if "reward" in result.get("history", [{}])[-1]:
        return result["history"][-1]["reward"]

    # Get the requested metric
    if metric_type == "llm_score":
        return metrics.get("llm_score", 0.0)
    elif metric_type == "f1_score":
        return metrics.get("f1_score", 0.0)
    elif metric_type == "bleu_score":
        return metrics.get("bleu_score", 0.0)
    else:
        # Default: average of three metrics
        f1 = metrics.get("f1_score", 0.0)
        bleu = metrics.get("bleu_score", 0.0)
        llm = metrics.get("llm_score", 0.0)
        return (f1 + bleu + llm) / 3.0


def calculate_replay_metrics(
    replay_dir: Path,
    metric_type: str = "llm_score"
) -> Dict:
    """
    Calculate two core metrics for Replay mode.

    Args:
        replay_dir: replay experiment directory (e.g., outputs/replay/mem0/seed66-1)
        metric_type: performance metric type

    Returns:
        dict containing Average Success Rate and Forgetting Gain
    """
    print(f"Reading Replay data directory: {replay_dir}")

    # 1. Find all replay stages
    replay_stages = sorted([d for d in replay_dir.iterdir() if d.is_dir() and d.name.startswith("replay")])
    K = len(replay_stages)
    print(f"Found {K} replay stages: {[s.name for s in replay_stages]}")

    # 2. Data structures
    # immediate_performance[stage_idx][(task, index)] = performance
    immediate_performance = {}
    # replay_performance[stage_idx][(task, index)] = performance
    replay_performance = defaultdict(dict)
    # stage_samples[stage_idx] = set of (task, index) learned in this stage
    stage_samples = defaultdict(set)

    # 3. Read all immediate test results from main directory test/
    immediate_test_dir = replay_dir / "test"
    print(f"\nReading immediate test results (main test/):")
    if immediate_test_dir.exists():
        for task_dir in immediate_test_dir.iterdir():
            if task_dir.is_dir():
                task_name = task_dir.name
                for sample_file in task_dir.glob("*.json"):
                    index = int(sample_file.stem)
                    result = load_sample_result(sample_file)
                    if result and result.get("split") == "immediate_test":
                        perf = get_performance(result, metric_type)
                        sample_key = (task_name, index)
                        # Store immediate test result (stage assignment determined later)
                        if "immediate_all" not in immediate_performance:
                            immediate_performance["immediate_all"] = {}
                        immediate_performance["immediate_all"][sample_key] = perf
                        print(f"  {task_name}/{index}: immediate={perf:.4f}")

    print(f"\nTotal immediate test results: {len(immediate_performance.get('immediate_all', {}))} samples")

    # 4. Determine which replay stage each sample was learned in using execution_order
    main_exec_order = replay_dir / "execution_order.json"
    sample_to_replay_stage = {}  # (task, index) -> replay_stage_idx

    if main_exec_order.exists():
        with open(main_exec_order, 'r', encoding='utf-8') as f:
            exec_data = json.load(f)
            # Infer stage assignment from replay1/test/execution_order
            replay1_test_exec = replay_stages[0] / "test" / "execution_order.json"
            if replay1_test_exec.exists():
                with open(replay1_test_exec, 'r', encoding='utf-8') as rf:
                    first_replay_samples = json.load(rf)
                    samples_per_stage = len(first_replay_samples)
                    print(f"\nApprox. samples per replay stage: {samples_per_stage}")

                    # Assign samples to stages based on execution_order
                    for i, item in enumerate(exec_data):
                        if item.get("split") == "train":
                            stage_idx = (i // samples_per_stage) + 1
                            sample_key = (item["task"], item["index"])
                            sample_to_replay_stage[sample_key] = stage_idx
                            stage_samples[stage_idx].add(sample_key)

    print(f"\nSuccessfully mapped {len(sample_to_replay_stage)} samples to replay stages")

    # 5. Read replay test results from each replay stage
    for stage_idx, stage_dir in enumerate(replay_stages, start=1):
        print(f"\nProcessing stage {stage_idx}: {stage_dir.name}")

        # 5.1 Read replay test results from test/
        test_dir = stage_dir / "test"
        if test_dir.exists():
            for task_dir in test_dir.iterdir():
                if task_dir.is_dir():
                    task_name = task_dir.name
                    for sample_file in task_dir.glob("*.json"):
                        index = int(sample_file.stem)
                        result = load_sample_result(sample_file)
                        if result and result.get("split") == "test":
                            perf = get_performance(result, metric_type)
                            sample_key = (task_name, index)
                            replay_performance[stage_idx][sample_key] = perf
                            print(f"  Replay test - {task_name}/{index}: replay={perf:.4f}")

    # 6. Calculate Average Success Rate
    all_replay_scores = []
    for stage_idx in replay_performance:
        all_replay_scores.extend(replay_performance[stage_idx].values())

    avg_success_rate = sum(all_replay_scores) / len(all_replay_scores) if all_replay_scores else 0.0

    print(f"\n{'='*60}")
    print(f"Average Success Rate: {avg_success_rate:.4f} ({avg_success_rate*100:.2f}%)")
    print(f"  (based on {len(all_replay_scores)} replay tests)")

    # 7. Calculate Forgetting Gain
    # FG_k^{(j)}(s) = (P_immediate^{(j)}(s) - P_replay_k^{(j)}(s)) / P_immediate^{(j)}(s) * 100%
    # FG = 1/(K-1) * Σ_{j=1}^{K-1} 1/|S_j| * Σ_{s∈S_j} 1/(K-j) * Σ_{k=j+1}^K FG_k^{(j)}(s)

    total_fg = 0.0
    num_valid_stages = 0

    print(f"\n{'='*60}")
    print("Calculating Forgetting Gain:")

    immediate_all = immediate_performance.get("immediate_all", {})

    for j in range(1, K):  # Iterate over all learning stages (except the last)
        samples_in_stage_j = stage_samples[j]
        if not samples_in_stage_j:
            continue

        stage_fg_sum = 0.0
        num_samples_with_replay = 0

        for sample in samples_in_stage_j:
            # Get the immediate test performance for this sample (from main test/)
            p_immediate = immediate_all.get(sample)
            if p_immediate is None or p_immediate == 0:
                continue

            # Calculate average forgetting over all subsequent replay stages
            sample_fg_sum = 0.0
            num_future_replays = 0

            for k in range(j + 1, K + 1):  # Iterate over all subsequent replay stages
                p_replay = replay_performance[k].get(sample)
                if p_replay is not None:
                    # Calculate forgetting
                    fg = ((p_immediate - p_replay) / p_immediate) * 100.0
                    sample_fg_sum += fg
                    num_future_replays += 1
                    print(f"  Stage {j} sample {sample}, tested at stage {k}: immediate={p_immediate:.4f}, replay={p_replay:.4f}, FG={fg:.2f}%")

            if num_future_replays > 0:
                # Average over all subsequent replays for this sample
                avg_sample_fg = sample_fg_sum / num_future_replays
                stage_fg_sum += avg_sample_fg
                num_samples_with_replay += 1

        if num_samples_with_replay > 0:
            # Average over all samples in this stage
            avg_stage_fg = stage_fg_sum / num_samples_with_replay
            total_fg += avg_stage_fg
            num_valid_stages += 1
            print(f"  Stage {j} avg forgetting: {avg_stage_fg:.2f}%")

    # Average over all valid stages
    overall_fg = total_fg / num_valid_stages if num_valid_stages > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Overall Forgetting Gain: {overall_fg:.2f}%")
    print(f"  (based on {num_valid_stages} learning stages)")

    # 8. Return results
    result = {
        "metric_type": metric_type,
        "num_replay_stages": K,
        "average_success_rate": avg_success_rate,
        "forgetting_gain": overall_fg,
        "num_replay_tests": len(all_replay_scores),
        "num_valid_stages": num_valid_stages
    }

    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python calculate_replay_metrics.py <replay_dir> [metric_type]")
        print("Example: python calculate_replay_metrics.py outputs/replay/mem0/seed66-1 llm_score")
        print("Metric types: llm_score (default), f1_score, bleu_score, average")
        sys.exit(1)

    replay_dir = Path(sys.argv[1])
    metric_type = sys.argv[2] if len(sys.argv) > 2 else "llm_score"

    if not replay_dir.exists():
        print(f"Error: directory does not exist - {replay_dir}")
        sys.exit(1)

    # Calculate metrics
    result = calculate_replay_metrics(replay_dir, metric_type)

    # Save results
    output_file = replay_dir / f"replay_metrics_{metric_type}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

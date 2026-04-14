#!/usr/bin/env python3
"""
Calculate metrics for Repair mode.
"""
import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_sample_result(file_path: Path) -> Dict:
    """Load the result file for a single sample."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data
    except Exception as e:
        print(f"Warning: Failed to load {file_path}: {e}")
        return None


def get_performance(result: Dict, metric_type: str = "llm_score") -> float:
    """Extract the specified metric from the result."""
    if not result:
        return None

    try:
        # Metrics are stored under result.result.metrics
        metrics = result.get("result", {}).get("result", {}).get("metrics", {})
        return metrics.get(metric_type, 0.0)
    except Exception as e:
        print(f"Warning: Failed to extract metric: {e}")
        return None


def calculate_repair_metrics(repair_dir: Path, metric_type: str = "llm_score") -> Dict:
    """
    Calculate metrics for Repair mode.

    Args:
        repair_dir: repair directory path (e.g., outputs/repair/mem0/seed66-1)
        metric_type: metric type (llm_score, f1_score, bleu_score)

    Returns:
        dict containing the following metrics:
        - repair_gain_full: repair gain across all samples
        - repair_gain_standard: repair gain for reversed samples
        - num_repair_stages: number of repair stages
        - avg_wrong_full: avg performance after wrong learning (all samples)
        - avg_right_full: avg performance after repair (all samples)
        - avg_wrong_standard: avg performance after wrong learning (reversed samples)
        - avg_right_standard: avg performance after repair (reversed samples)
    """
    repair_dir = Path(repair_dir)

    # 1. Find all repair stage directories
    repair_stages = sorted([d for d in repair_dir.iterdir() if d.is_dir() and d.name.startswith("repair")],
                          key=lambda x: int(x.name.replace("repair", "")))

    K = len(repair_stages)
    print(f"\n{'='*60}")
    print(f"Processing directory: {repair_dir}")
    print(f"Found {K} repair stages")

    # 2. Collect performance data across all stages
    all_wrong_full_scores = []
    all_right_full_scores = []
    all_wrong_standard_scores = []
    all_right_standard_scores = []

    for stage_idx, stage_dir in enumerate(repair_stages, start=1):
        print(f"\nProcessing stage {stage_idx}: {stage_dir.name}")

        # Read results from the four test directories
        for test_type in ["wrongJudgeTestFull", "rightJudgeTestFull",
                         "wrongJudgeTestStandard", "rightJudgeTestStandard"]:
            test_dir = stage_dir / test_type

            if not test_dir.exists():
                print(f"  Warning: {test_type} directory not found")
                continue

            stage_scores = []

            # Iterate over all task directories
            for task_dir in test_dir.iterdir():
                if task_dir.is_dir():
                    task_name = task_dir.name

                    # Read all sample results
                    for sample_file in task_dir.glob("*.json"):
                        result = load_sample_result(sample_file)
                        if result:
                            perf = get_performance(result, metric_type)
                            if perf is not None:
                                stage_scores.append(perf)

            # Store in the corresponding list
            if test_type == "wrongJudgeTestFull":
                all_wrong_full_scores.extend(stage_scores)
                print(f"  {test_type}: {len(stage_scores)} samples, avg {sum(stage_scores)/len(stage_scores):.4f}")
            elif test_type == "rightJudgeTestFull":
                all_right_full_scores.extend(stage_scores)
                print(f"  {test_type}: {len(stage_scores)} samples, avg {sum(stage_scores)/len(stage_scores):.4f}")
            elif test_type == "wrongJudgeTestStandard":
                all_wrong_standard_scores.extend(stage_scores)
                print(f"  {test_type}: {len(stage_scores)} samples, avg {sum(stage_scores)/len(stage_scores):.4f}")
            elif test_type == "rightJudgeTestStandard":
                all_right_standard_scores.extend(stage_scores)
                print(f"  {test_type}: {len(stage_scores)} samples, avg {sum(stage_scores)/len(stage_scores):.4f}")

    # 3. Compute average performance
    avg_wrong_full = sum(all_wrong_full_scores) / len(all_wrong_full_scores) if all_wrong_full_scores else 0.0
    avg_right_full = sum(all_right_full_scores) / len(all_right_full_scores) if all_right_full_scores else 0.0
    avg_wrong_standard = sum(all_wrong_standard_scores) / len(all_wrong_standard_scores) if all_wrong_standard_scores else 0.0
    avg_right_standard = sum(all_right_standard_scores) / len(all_right_standard_scores) if all_right_standard_scores else 0.0

    # 4. Compute Repair Gain
    repair_gain_full = avg_right_full - avg_wrong_full
    repair_gain_standard = avg_right_standard - avg_wrong_standard

    print(f"\n{'='*60}")
    print(f"Repair performance statistics:")
    print(f"\n[Full (all samples)]")
    print(f"  Accuracy after wrong learning (full):  {avg_wrong_full:.4f} ({avg_wrong_full*100:.2f}%)")
    print(f"  Accuracy after repair (full):          {avg_right_full:.4f} ({avg_right_full*100:.2f}%)")
    print(f"  Repair gain (full):                    {repair_gain_full:.4f} ({repair_gain_full*100:.2f}%)")

    print(f"\n[Standard (reversed samples)]")
    print(f"  Accuracy after wrong learning (std):   {avg_wrong_standard:.4f} ({avg_wrong_standard*100:.2f}%)")
    print(f"  Accuracy after repair (std):           {avg_right_standard:.4f} ({avg_right_standard*100:.2f}%)")
    print(f"  Repair gain (std):                     {repair_gain_standard:.4f} ({repair_gain_standard*100:.2f}%)")

    print(f"\nSample counts:")
    print(f"  wrongJudgeTestFull: {len(all_wrong_full_scores)} samples")
    print(f"  rightJudgeTestFull: {len(all_right_full_scores)} samples")
    print(f"  wrongJudgeTestStandard: {len(all_wrong_standard_scores)} samples")
    print(f"  rightJudgeTestStandard: {len(all_right_standard_scores)} samples")

    # 5. Return results
    result = {
        "metric_type": metric_type,
        "num_repair_stages": K,
        "repair_gain_full": repair_gain_full,
        "repair_gain_standard": repair_gain_standard,
        "avg_wrong_full": avg_wrong_full,
        "avg_right_full": avg_right_full,
        "avg_wrong_standard": avg_wrong_standard,
        "avg_right_standard": avg_right_standard,
        "num_samples_full": len(all_wrong_full_scores),
        "num_samples_standard": len(all_wrong_standard_scores)
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Calculate Repair mode metrics")
    parser.add_argument("repair_dir", type=str, help="Repair directory path")
    parser.add_argument("--metric", type=str, default="llm_score",
                       choices=["llm_score", "f1_score", "bleu_score"],
                       help="Metric type (default: llm_score)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file path (optional)")

    args = parser.parse_args()

    # Calculate metrics
    result = calculate_repair_metrics(args.repair_dir, args.metric)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(args.repair_dir) / f"repair_metrics_{args.metric}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

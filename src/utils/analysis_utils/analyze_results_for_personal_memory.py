"""
Script to analyze personal memory (locomo) benchmark run results.

Usage:
    python -m src.utils.analyze_results_for_personal_memory outputs/2025-12-28_12-58-09/locomo-0
"""

import json
import sys
import io
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set Windows console encoding to UTF-8
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass


def analyze_results(result_dir: Path) -> Dict[str, Any]:
    """Analyze all JSON files in the result directory."""
    # Get all JSON files (excluding .error.json files)
    all_json_files = list(result_dir.glob("*.json"))

    # Separate regular JSON files and .error.json files
    normal_files = []
    error_files = []

    for p in all_json_files:
        if p.stem.endswith(".error"):
            # Extract the numeric part (e.g. "251.error" -> 251)
            try:
                num = int(p.stem.split(".")[0])
                error_files.append((num, p))
            except ValueError:
                continue
        else:
            # Regular JSON file
            try:
                num = int(p.stem)
                normal_files.append((num, p))
            except ValueError:
                continue

    # Sort by number
    json_files = [p for _, p in sorted(normal_files, key=lambda x: x[0])]

    if not json_files:
        print(f"[ERROR] No JSON files found in: {result_dir}")
        return {}

    # Stats data structure
    stats = {
        "total_samples": len(json_files),
        "failed_samples": len(error_files),
        "task_name": None,
        "agent_name": None,
        # Overall metrics
        "f1_scores": [],
        "bleu_scores": [],
        "llm_scores": [],
        # Grouped by category
        "by_category": defaultdict(lambda: {
            "count": 0,
            "f1_scores": [],
            "bleu_scores": [],
            "llm_scores": [],
        }),
    }

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARNING] Cannot read {json_file.name}: {e}")
            continue

        result = data.get("result", {})
        index = result.get("index", data.get("index", json_file.stem))

        # Extract basic info
        if stats["task_name"] is None:
            stats["task_name"] = result.get("task", data.get("task", "unknown"))
        if stats["agent_name"] is None:
            stats["agent_name"] = result.get("agent_name", "unknown")

        # Check status (status may be in result or at top level)
        status = result.get("status") or data.get("status", "")
        if status != "completed":
            continue

        # Extract metrics
        metrics = result.get("metrics", {})
        if not metrics:
            continue

        f1_score = metrics.get("f1_score")
        bleu_score = metrics.get("bleu_score")
        llm_score = metrics.get("llm_score")

        # Only count valid scores (between 0 and 1)
        if f1_score is not None and 0 <= f1_score <= 1:
            stats["f1_scores"].append(f1_score)
        if bleu_score is not None and 0 <= bleu_score <= 1:
            stats["bleu_scores"].append(bleu_score)
        if llm_score is not None and 0 <= llm_score <= 1:
            stats["llm_scores"].append(llm_score)

        # Group stats by category
        category = result.get("category")
        if category is not None:
            cat_key = f"category_{category}"
            if f1_score is not None and 0 <= f1_score <= 1:
                stats["by_category"][cat_key]["f1_scores"].append(f1_score)
            if bleu_score is not None and 0 <= bleu_score <= 1:
                stats["by_category"][cat_key]["bleu_scores"].append(bleu_score)
            if llm_score is not None and 0 <= llm_score <= 1:
                stats["by_category"][cat_key]["llm_scores"].append(llm_score)
            stats["by_category"][cat_key]["count"] += 1

    # Compute overall averages
    stats["avg_f1_score"] = sum(stats["f1_scores"]) / len(stats["f1_scores"]) if stats["f1_scores"] else 0.0
    stats["avg_bleu_score"] = sum(stats["bleu_scores"]) / len(stats["bleu_scores"]) if stats["bleu_scores"] else 0.0
    stats["avg_llm_score"] = sum(stats["llm_scores"]) / len(stats["llm_scores"]) if stats["llm_scores"] else 0.0

    # Compute per-category averages
    for cat_key, cat_stats in stats["by_category"].items():
        cat_stats["avg_f1_score"] = sum(cat_stats["f1_scores"]) / len(cat_stats["f1_scores"]) if cat_stats["f1_scores"] else 0.0
        cat_stats["avg_bleu_score"] = sum(cat_stats["bleu_scores"]) / len(cat_stats["bleu_scores"]) if cat_stats["bleu_scores"] else 0.0
        cat_stats["avg_llm_score"] = sum(cat_stats["llm_scores"]) / len(cat_stats["llm_scores"]) if cat_stats["llm_scores"] else 0.0

    return stats


def print_report(stats: Dict[str, Any], result_dir: Path):
    """Print the analysis report."""
    print("=" * 80)
    print("Personal Memory (Locomo) Benchmark Results Report")
    print("=" * 80)
    print(f"\nResult directory: {result_dir}")
    print(f"Task: {stats.get('task_name', 'unknown')}")
    print(f"Agent: {stats.get('agent_name', 'unknown')}")

    print(f"\n{'─' * 80}")
    print("Overall Statistics")
    print(f"{'─' * 80}")
    total = stats["total_samples"]
    failed = stats["failed_samples"]
    print(f"Total samples: {total}")
    print(f"Failed samples: {failed}")
    print(f"Valid samples: {total - failed}")

    print(f"\n{'─' * 80}")
    print("Overall Average Metrics")
    print(f"{'─' * 80}")
    print(f"Avg F1 Score:   {stats['avg_f1_score']:.4f}")
    print(f"Avg BLEU Score: {stats['avg_bleu_score']:.4f}")
    print(f"Avg LLM Score:  {stats['avg_llm_score']:.4f}")

    # Stats grouped by category
    if stats["by_category"]:
        print(f"\n{'─' * 80}")
        print("Stats by Category")
        print(f"{'─' * 80}")

        # Sort categories numerically
        sorted_categories = sorted(
            stats["by_category"].items(),
            key=lambda x: int(x[0].split("_")[1]) if x[0].startswith("category_") else 0
        )

        for cat_key, cat_stats in sorted_categories:
            category_num = cat_key.split("_")[1] if cat_key.startswith("category_") else "?"
            print(f"\nCategory {category_num}:")
            print(f"  Samples: {cat_stats['count']}")
            print(f"  Avg F1 Score:   {cat_stats['avg_f1_score']:.4f}")
            print(f"  Avg BLEU Score: {cat_stats['avg_bleu_score']:.4f}")
            print(f"  Avg LLM Score:  {cat_stats['avg_llm_score']:.4f}")

    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.analyze_results_for_personal_memory <result_dir>")
        print("Example: python -m src.utils.analyze_results_for_personal_memory outputs/2025-12-28_12-58-09/locomo-0")
        sys.exit(1)

    result_dir = Path(sys.argv[1])
    if not result_dir.exists():
        print(f"[ERROR] Directory does not exist: {result_dir}")
        sys.exit(1)

    stats = analyze_results(result_dir)
    if stats:
        print_report(stats, result_dir)
    else:
        print("[ERROR] Failed to analyze any results")


if __name__ == "__main__":
    main()

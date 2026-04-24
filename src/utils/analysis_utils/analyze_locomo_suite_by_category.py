"""
Analyze a full LoCoMo result suite and report weighted metrics by category.

Usage:
    python -m src.utils.analysis_utils.analyze_locomo_suite_by_category outputs/qwen3-32B-zero-shot-locomo
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.utils.analysis_utils.analyze_results_for_personal_memory import analyze_results


def _iter_locomo_result_dirs(base_dir: Path) -> Iterable[Path]:
    for outer_dir in sorted(base_dir.iterdir()):
        if not outer_dir.is_dir():
            continue

        inner_dirs = [p for p in outer_dir.iterdir() if p.is_dir() and p.name.startswith("locomo-")]
        if not inner_dirs:
            continue

        yield inner_dirs[0]


def analyze_locomo_suite(base_dir: Path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "base_dir": str(base_dir),
        "num_runs": 0,
        "total_valid_samples": 0,
        "by_category": defaultdict(
            lambda: {
                "count": 0,
                "f1_scores": [],
                "bleu_scores": [],
                "llm_scores": [],
            }
        ),
    }

    for result_dir in _iter_locomo_result_dirs(base_dir):
        run_stats = analyze_results(result_dir)
        if not run_stats:
            continue

        stats["num_runs"] += 1

        for cat_key, cat_stats in run_stats["by_category"].items():
            stats["by_category"][cat_key]["count"] += cat_stats["count"]
            stats["by_category"][cat_key]["f1_scores"].extend(cat_stats["f1_scores"])
            stats["by_category"][cat_key]["bleu_scores"].extend(cat_stats["bleu_scores"])
            stats["by_category"][cat_key]["llm_scores"].extend(cat_stats["llm_scores"])
            stats["total_valid_samples"] += cat_stats["count"]

    for cat_key, cat_stats in stats["by_category"].items():
        cat_stats["avg_f1_score"] = (
            sum(cat_stats["f1_scores"]) / len(cat_stats["f1_scores"]) if cat_stats["f1_scores"] else 0.0
        )
        cat_stats["avg_bleu_score"] = (
            sum(cat_stats["bleu_scores"]) / len(cat_stats["bleu_scores"]) if cat_stats["bleu_scores"] else 0.0
        )
        cat_stats["avg_llm_score"] = (
            sum(cat_stats["llm_scores"]) / len(cat_stats["llm_scores"]) if cat_stats["llm_scores"] else 0.0
        )

    all_f1: List[float] = []
    all_bleu: List[float] = []
    all_llm: List[float] = []
    for cat_stats in stats["by_category"].values():
        all_f1.extend(cat_stats["f1_scores"])
        all_bleu.extend(cat_stats["bleu_scores"])
        all_llm.extend(cat_stats["llm_scores"])

    stats["overall_avg_f1_score"] = sum(all_f1) / len(all_f1) if all_f1 else 0.0
    stats["overall_avg_bleu_score"] = sum(all_bleu) / len(all_bleu) if all_bleu else 0.0
    stats["overall_avg_llm_score"] = sum(all_llm) / len(all_llm) if all_llm else 0.0
    return stats


def print_report(stats: Dict[str, Any]) -> None:
    print("=" * 80)
    print("LoCoMo Suite Category Report")
    print("=" * 80)
    print(f"Base directory: {stats['base_dir']}")
    print(f"Detected runs: {stats['num_runs']}")
    print(f"Total valid samples: {stats['total_valid_samples']}")

    print("\nOverall Weighted Average")
    print("-" * 80)
    print(f"F1:   {stats['overall_avg_f1_score']:.4f}")
    print(f"BLEU: {stats['overall_avg_bleu_score']:.4f}")
    print(f"LLM:  {stats['overall_avg_llm_score']:.4f}")

    print("\nWeighted Average by Category")
    print("-" * 80)
    sorted_categories = sorted(
        stats["by_category"].items(),
        key=lambda x: int(x[0].split("_")[1]) if x[0].startswith("category_") else 0,
    )
    for cat_key, cat_stats in sorted_categories:
        category_num = cat_key.split("_")[1] if cat_key.startswith("category_") else "?"
        print(
            f"Category {category_num}: "
            f"count={cat_stats['count']}  "
            f"F1={cat_stats['avg_f1_score']:.4f}  "
            f"BLEU={cat_stats['avg_bleu_score']:.4f}  "
            f"LLM={cat_stats['avg_llm_score']:.4f}"
        )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.analysis_utils.analyze_locomo_suite_by_category <suite_dir>")
        print("Example: python -m src.utils.analysis_utils.analyze_locomo_suite_by_category outputs/qwen3-32B-zero-shot-locomo")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    if not base_dir.exists():
        print(f"[ERROR] Directory does not exist: {base_dir}")
        sys.exit(1)

    stats = analyze_locomo_suite(base_dir)
    if stats["num_runs"] == 0:
        print(f"[ERROR] No locomo result directories found under: {base_dir}")
        sys.exit(1)

    print_report(stats)


if __name__ == "__main__":
    main()

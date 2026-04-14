"""
Script to analyze benchmark run results.

Usage:
    python -m src.utils.analyze_results outputs/2025-12-05_19-45-36/dbbench-std
"""

import json
import sys
import io
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any

# Set Windows console encoding to UTF-8
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except Exception:
        pass


def count_turns(history: List[Dict[str, Any]]) -> int:
    """Count the number of conversation turns (number of assistant messages)."""
    return sum(1 for msg in history if msg.get("role") == "assistant")


def extract_error_type(error_msg: str) -> str:
    """Extract the error type from an error message."""
    if not error_msg:
        return "None"
    error_lower = error_msg.lower()
    if "timeout" in error_lower or "timed out" in error_lower:
        return "Timeout"
    elif "400" in error_msg or "bad request" in error_lower:
        return "400 Bad Request"
    elif "429" in error_msg or "too many requests" in error_lower:
        return "429 Rate Limit"
    elif "500" in error_msg or "internal server error" in error_lower or "upstream error" in error_lower:
        return "500 Server Error"
    elif "connection" in error_lower or "connection aborted" in error_lower or "connectionreset" in error_lower:
        return "Connection Error"
    else:
        return "Other Error"


def analyze_error_cause(error_msg: str, history: List[Dict[str, Any]]) -> str:
    """Analyze the root cause of an error."""
    error_lower = error_msg.lower()
    turn_count = count_turns(history)

    if "timeout" in error_lower or "timed out" in error_lower:
        if turn_count == 0:
            return "First-call timeout - LLM API response exceeded threshold (possibly high server load or slow first request)"
        else:
            return f"Turn {turn_count+1} timeout - possibly context too long or slow server response"
    elif "connection" in error_lower or "connection aborted" in error_lower:
        if "max retries exceeded" in error_lower:
            return "Connection failed - unable to reach API server (network issue, DNS failure, or service unavailable)"
        elif "connection aborted" in error_lower or "connectionreset" in error_lower:
            return "Connection interrupted - remote host forcibly closed the connection (server dropped or network instability)"
        else:
            return "Connection error - network connectivity issue"
    elif "500" in error_msg or "upstream error" in error_lower:
        return "Server error - API server internal failure (upstream service failed; retries implemented but persistent failure may still occur)"
    elif "400" in error_msg:
        return "Request error - client request format issue (validation implemented but edge cases may still occur)"
    else:
        return "Unknown error"


def analyze_results(result_dir: Path) -> Dict[str, Any]:
    """Analyze all JSON files in the result directory."""
    # Get all JSON files, including .error.json files
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
                # Skip if numeric part cannot be extracted
                continue
        else:
            # Regular JSON file
            try:
                num = int(p.stem)
                normal_files.append((num, p))
            except ValueError:
                # Skip if stem cannot be converted to int
                continue

    # Sort by number
    json_files = [p for _, p in sorted(normal_files + error_files, key=lambda x: x[0])]

    if not json_files:
        print(f"[ERROR] No JSON files found in: {result_dir}")
        return {}

    stats = {
        "total_samples": len(json_files),
        "completed": 0,
        "failed": 0,
        "reward_1": 0,
        "reward_0": 0,
        "no_reward": 0,
        "error_types": Counter(),
        "turn_counts": [],
        "error_samples": [],
        "success_samples": [],
        "task_name": None,
        "agent_name": None,
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

        # Count status
        status = result.get("status", "")
        if status == "completed":
            stats["completed"] += 1
        elif result.get("error"):
            stats["failed"] += 1

        # Count reward
        reward = result.get("reward")
        if reward == 1:
            stats["reward_1"] += 1
            stats["success_samples"].append(index)
        elif reward == 0:
            stats["reward_0"] += 1
        else:
            stats["no_reward"] += 1

        # Count error types
        error = result.get("error")
        if error:
            error_type = extract_error_type(error)
            error_cause = analyze_error_cause(error, history)
            stats["error_types"][error_type] += 1
            stats["error_samples"].append({
                "index": index,
                "error_type": error_type,
                "error_cause": error_cause,
                "turn_count": count_turns(history),
                "error": error[:200] + "..." if len(error) > 200 else error,
            })

        # Count turns
        history = data.get("history", [])
        if history:
            turns = count_turns(history)
            stats["turn_counts"].append(turns)

    # Compute average turns
    if stats["turn_counts"]:
        stats["avg_turns"] = sum(stats["turn_counts"]) / len(stats["turn_counts"])
        stats["min_turns"] = min(stats["turn_counts"])
        stats["max_turns"] = max(stats["turn_counts"])
    else:
        stats["avg_turns"] = 0
        stats["min_turns"] = 0
        stats["max_turns"] = 0

    return stats


def print_report(stats: Dict[str, Any], result_dir: Path):
    """Print the analysis report."""
    print("=" * 80)
    print("Benchmark Results Report")
    print("=" * 80)
    print(f"\nResult directory: {result_dir}")
    print(f"Task: {stats.get('task_name', 'unknown')}")
    print(f"Agent: {stats.get('agent_name', 'unknown')}")

    print(f"\n{'─' * 80}")
    print("Overall Statistics")
    print(f"{'─' * 80}")
    total = stats["total_samples"]
    print(f"Total samples: {total}")
    print(f"Completed: {stats['completed']} ({stats['completed']/total*100:.1f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/total*100:.1f}%)")

    print(f"\n{'─' * 80}")
    print("Reward Distribution")
    print(f"{'─' * 80}")
    print(f"Reward = 1 (success): {stats['reward_1']} ({stats['reward_1']/total*100:.1f}%)")
    print(f"Reward = 0 (failure): {stats['reward_0']} ({stats['reward_0']/total*100:.1f}%)")
    print(f"No reward: {stats['no_reward']} ({stats['no_reward']/total*100:.1f}%)")

    if stats["turn_counts"]:
        print(f"\n{'─' * 80}")
        print("Turn Count Statistics")
        print(f"{'─' * 80}")
        print(f"Avg turns: {stats['avg_turns']:.2f}")
        print(f"Min turns: {stats['min_turns']}")
        print(f"Max turns: {stats['max_turns']}")

    if stats["error_types"]:
        print(f"\n{'─' * 80}")
        print("Error Type Distribution")
        print(f"{'─' * 80}")
        for error_type, count in stats["error_types"].most_common():
            print(f"  {error_type}: {count} ({count/total*100:.1f}%)")

    if stats["error_samples"]:
        print(f"\n{'─' * 80}")
        print("Error Sample Details (first 10)")
        print(f"{'─' * 80}")
        for i, sample in enumerate(stats["error_samples"][:10], 1):
            print(f"\n  {i}. Index {sample['index']} - {sample['error_type']}")
            print(f"     Turns: {sample.get('turn_count', 0)}")
            print(f"     Cause: {sample.get('error_cause', 'Unknown')}")
            print(f"     Error: {sample['error']}")
        if len(stats["error_samples"]) > 10:
            print(f"\n  ... {len(stats['error_samples']) - 10} more error samples")

    print("\n" + "=" * 80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.utils.analyze_results <result_dir>")
        print("Example: python -m src.utils.analyze_results outputs/2025-12-05_19-45-36/dbbench-std")
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

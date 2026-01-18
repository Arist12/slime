import argparse
from typing import Any
from tqdm import tqdm
from utils import load_from_jsonl, save_to_jsonl, normalize_code, code_edit_grader


def grade_code_edit_benchmark(result_file: str) -> dict[str, Any]:
    """Grade code edit benchmark results that can handle all types of benchmarks."""

    # Load results
    print(f"Loading results from {result_file}...")
    results = load_from_jsonl(result_file)

    stats = {
        "total_samples": 0,
        "normalized_matches": 0,
        "find_replace_attempts": 0,
        "find_replace_normalized_matches": 0,
        "find_replace_format_correct": 0,
        "fully_rewrite_attempts": 0,
        "fully_rewrite_normalized_matches": 0,
        "fully_rewrite_format_correct": 0,
        "format_failures": 0,
    }

    for result in tqdm(results, desc="Grading samples"):
        stats["total_samples"] += 1

        # Extract basic information
        original_code = result["original_code"]
        ground_truth = result["ground_truth"]
        model_response = result["model_response"]

        if "</think>" in model_response:
            model_response = model_response.split("</think>")[1]

        # Determine edit mode and extract result
        edit_mode, format_success, extracted_code = code_edit_grader(model_response, original_code)

        # Calculate matches
        if format_success:
            normalized_match = normalize_code(extracted_code) == normalize_code(ground_truth)
        else:
            normalized_match = False

        ######### Add grading results to the result object ########
        result["edit_mode"] = edit_mode
        result["format_success"] = format_success
        result["normalized_match"] = normalized_match
        result["extracted_code"] = extracted_code
        ###########################################################

        # Update statistics
        if normalized_match:
            stats["normalized_matches"] += 1

        if edit_mode == "find_replace":
            stats["find_replace_attempts"] += 1
            if format_success:
                stats["find_replace_format_correct"] += 1
            if normalized_match:
                stats["find_replace_normalized_matches"] += 1

        elif edit_mode == "fully_rewrite":
            stats["fully_rewrite_attempts"] += 1
            if format_success:
                stats["fully_rewrite_format_correct"] += 1
            if normalized_match:
                stats["fully_rewrite_normalized_matches"] += 1

        if not format_success:
            stats["format_failures"] += 1

    return results, stats


def print_statistics(stats: dict[str, Any], file_path: str):
    """Print the statistics in a clear format matching analyze_results.py."""
    print("\n" + "="*60)
    print(f"ANALYSIS RESULTS FOR: {file_path}")
    print("="*60)

    total = stats["total_samples"]
    if total == 0:
        print("No samples found!")
        return

    # Overall statistics
    normalized_acc = stats["normalized_matches"] / total

    print(f"Total samples: {total}")
    print(f"Overall normalized match accuracy: {normalized_acc:.4f} ({stats['normalized_matches']}/{total})")

    print("\n" + "-"*40)
    print("FIND-REPLACE ANALYSIS")
    print("-"*40)

    fr_attempts = stats["find_replace_attempts"]
    if fr_attempts > 0:
        fr_rate = fr_attempts / total
        fr_normalized_rate = stats["find_replace_normalized_matches"] / fr_attempts
        fr_format_rate = stats["find_replace_format_correct"] / fr_attempts

        print(f"Find-replace attempts: {fr_rate:.4f} ({fr_attempts}/{total})")
        print(f"Find-replace normalized match rate: {fr_normalized_rate:.4f} ({stats['find_replace_normalized_matches']}/{fr_attempts})")
        print(f"Find-replace format correct rate: {fr_format_rate:.4f} ({stats['find_replace_format_correct']}/{fr_attempts})")
    else:
        print("No find-replace attempts found.")

    print("\n" + "-"*40)
    print("FULLY REWRITE ANALYSIS")
    print("-"*40)

    fw_attempts = stats["fully_rewrite_attempts"]
    if fw_attempts > 0:
        fw_rate = fw_attempts / total
        fw_normalized_rate = stats["fully_rewrite_normalized_matches"] / fw_attempts
        fw_format_rate = stats["fully_rewrite_format_correct"] / fw_attempts

        print(f"Fully rewrite attempts: {fw_rate:.4f} ({fw_attempts}/{total})")
        print(f"Fully rewrite normalized match rate: {fw_normalized_rate:.4f} ({stats['fully_rewrite_normalized_matches']}/{fw_attempts})")
        print(f"Fully rewrite format correct rate: {fw_format_rate:.4f} ({stats['fully_rewrite_format_correct']}/{fw_attempts})")
    else:
        print("No fully rewrite attempts found.")

    print("\n" + "-"*40)
    print("ERROR ANALYSIS")
    print("-"*40)

    format_failure_rate = stats["format_failures"] / total
    print(f"Format failure rate: {format_failure_rate:.4f} ({stats['format_failures']}/{total})")
    print(f"Format Success Rate: {1 - stats['format_failures'] / total:.4f} ({500 - stats['format_failures']}/{total})")


def main():
    parser = argparse.ArgumentParser(description="Grade unified benchmark evaluation results")
    parser.add_argument("result_file", help="Path to the result JSONL file")
    args = parser.parse_args()

    # Grade results
    graded_results, stats = grade_code_edit_benchmark(args.result_file)
    print_statistics(stats, args.result_file)
    save_to_jsonl(graded_results, args.result_file.replace(".jsonl", "_graded.jsonl"))
    print(f"\nGraded results saved to: {args.result_file.replace('.jsonl', '_graded.jsonl')}")

if __name__ == "__main__":
    main()

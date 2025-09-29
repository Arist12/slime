import argparse
import json
from typing import Any
from tqdm import tqdm
from utils import load_from_jsonl, save_to_jsonl, determine_format_and_success, normalize_code, calculate_token_consumption
from transformers import AutoTokenizer


def grade_unified_benchmark(results: list[dict[str, Any]], tokenizer: AutoTokenizer) -> dict[str, Any]:
    """Grade unified benchmark results that can handle all types of benchmarks."""
    stats = {
        "total_samples": 0,
        "exact_matches": 0,
        "normalized_matches": 0,
        "find_replace_attempts": 0,
        "find_replace_exact_matches": 0,
        "find_replace_normalized_matches": 0,
        "find_replace_format_correct": 0,
        "fully_rewrite_attempts": 0,
        "fully_rewrite_exact_matches": 0,
        "fully_rewrite_normalized_matches": 0,
        "fully_rewrite_format_correct": 0,
        "format_failures": 0,
        "total_tokens": 0
    }

    for result in tqdm(results, desc="Grading samples"):
        stats["total_samples"] += 1

        # Extract basic information
        original_code = result["original_code"]
        ground_truth = result["ground_truth"]
        model_response = result["model_response"]

        # Determine edit mode and extract result
        edit_mode, format_success, extracted_code = determine_format_and_success(model_response, original_code)

        # Calculate matches
        if format_success:
            exact_match = extracted_code == ground_truth
            normalized_match = normalize_code(extracted_code) == normalize_code(ground_truth)
        else:
            exact_match = False
            normalized_match = False

        token_consumption = calculate_token_consumption(tokenizer, model_response)

        # Add grading results to the result object
        result["edit_mode"] = edit_mode
        result["format_success"] = format_success
        result["exact_match"] = exact_match
        result["normalized_match"] = normalized_match
        result["token_consumption"] = token_consumption
        result["extracted_code"] = extracted_code

        # Update statistics
        if exact_match:
            stats["exact_matches"] += 1
        if normalized_match:
            stats["normalized_matches"] += 1

        stats["total_tokens"] += token_consumption

        if edit_mode == "find_replace":
            stats["find_replace_attempts"] += 1
            if format_success:
                stats["find_replace_format_correct"] += 1
            if exact_match:
                stats["find_replace_exact_matches"] += 1
            if normalized_match:
                stats["find_replace_normalized_matches"] += 1

        elif edit_mode == "fully_rewrite":
            stats["fully_rewrite_attempts"] += 1
            if format_success:
                stats["fully_rewrite_format_correct"] += 1
            if exact_match:
                stats["fully_rewrite_exact_matches"] += 1
            if normalized_match:
                stats["fully_rewrite_normalized_matches"] += 1

        else:
            stats["format_failures"] += 1

    return stats


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
    exact_acc = stats["exact_matches"] / total
    normalized_acc = stats["normalized_matches"] / total
    avg_tokens = stats["total_tokens"] / total

    print(f"Total samples: {total}")
    print(f"Overall exact match accuracy: {exact_acc:.4f} ({stats['exact_matches']}/{total})")
    print(f"Overall normalized match accuracy: {normalized_acc:.4f} ({stats['normalized_matches']}/{total})")
    print(f"Average token consumption: {avg_tokens:.2f}")

    print("\n" + "-"*40)
    print("FIND-REPLACE ANALYSIS")
    print("-"*40)

    fr_attempts = stats["find_replace_attempts"]
    if fr_attempts > 0:
        fr_rate = fr_attempts / total
        fr_exact_rate = stats["find_replace_exact_matches"] / fr_attempts
        fr_normalized_rate = stats["find_replace_normalized_matches"] / fr_attempts
        fr_format_rate = stats["find_replace_format_correct"] / fr_attempts

        print(f"Find-replace attempts: {fr_rate:.4f} ({fr_attempts}/{total})")
        print(f"Find-replace exact match rate: {fr_exact_rate:.4f} ({stats['find_replace_exact_matches']}/{fr_attempts})")
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
        fw_exact_rate = stats["fully_rewrite_exact_matches"] / fw_attempts
        fw_normalized_rate = stats["fully_rewrite_normalized_matches"] / fw_attempts
        fw_format_rate = stats["fully_rewrite_format_correct"] / fw_attempts

        print(f"Fully rewrite attempts: {fw_rate:.4f} ({fw_attempts}/{total})")
        print(f"Fully rewrite exact match rate: {fw_exact_rate:.4f} ({stats['fully_rewrite_exact_matches']}/{fw_attempts})")
        print(f"Fully rewrite normalized match rate: {fw_normalized_rate:.4f} ({stats['fully_rewrite_normalized_matches']}/{fw_attempts})")
        print(f"Fully rewrite format correct rate: {fw_format_rate:.4f} ({stats['fully_rewrite_format_correct']}/{fw_attempts})")
    else:
        print("No fully rewrite attempts found.")

    print("\n" + "-"*40)
    print("ERROR ANALYSIS")
    print("-"*40)

    format_failure_rate = stats["format_failures"] / total
    print(f"Format failure rate: {format_failure_rate:.4f} ({stats['format_failures']}/{total})")


def main():
    parser = argparse.ArgumentParser(description="Grade unified benchmark evaluation results")
    parser.add_argument("result_file", help="Path to the result JSONL file")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B"], help="Model name or path")
    args = parser.parse_args()

    try:
        # Load results
        print(f"Loading results from {args.result_file}...")
        results = load_from_jsonl(args.result_file)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        stats = grade_unified_benchmark(results, tokenizer)

        # Print statistics
        print_statistics(stats, args.result_file)

        # Save graded results if output path provided
        save_to_jsonl(results, args.result_file.replace(".jsonl", "_graded.jsonl"))
        print(f"\nGraded results saved to: {args.result_file.replace('.jsonl', '_graded.jsonl')}")

    except FileNotFoundError:
        print(f"Error: Could not find file '{args.result_file}'")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in file - {e}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()

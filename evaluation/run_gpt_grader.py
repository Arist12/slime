import argparse
import asyncio
import difflib
import os
from typing import Any

from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm
from prompt import SYSTEM_PROMPT, USER_PROMPT
from utils import load_from_jsonl, save_to_jsonl

load_dotenv()


def get_diffs(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    diffs = []

    for result in tqdm(results, desc="Getting diffs"):
        original_code = result["original_code"]
        ground_truth = result["ground_truth"]
        extracted_code = result["extracted_code"]

        item = {
            "original_code": original_code,
            "diff1": generate_diff(original_code, ground_truth),
            "diff2": generate_diff(original_code, extracted_code)
        }

        diffs.append(item)

    return diffs


def generate_diff(original_code: str, modified_code: str) -> str:
    """Generate a clean unified diff between original and modified code."""
    if not modified_code:
        return ""

    original_lines = original_code.splitlines(keepends=True)
    modified_lines = modified_code.splitlines(keepends=True)

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        lineterm=""
    )

    # Clean the diff - keep only meaningful lines
    diff_lines = []
    for line in list(diff)[2:]:  # Skip first two header lines
        if line.startswith(('@', '+', '-', ' ')):
            diff_lines.append(line.rstrip())

    return '\n'.join(diff_lines) if diff_lines else ""


def make_user_prompt(original_code: str, diff1: str, diff2: str) -> str:
    return USER_PROMPT.format(original_code=original_code, diff1=diff1, diff2=diff2)


def extract_result(llm_response: str) -> bool:
    llm_response = llm_response.upper().split("RESULT:")[1].strip()
    return 'NOT_EQUIVALENT' not in llm_response


async def evaluate_single_result(client: AsyncAzureOpenAI, semaphore: asyncio.Semaphore, result: dict, diff: dict):
    """Evaluate a single result asynchronously."""
    if result["normalized_match"]:
        result["gpt_eval_match"] = True
        result["gpt_eval_analysis"] = "Rule-based matching"
        return

    if diff["diff1"] == "" or diff["diff2"] == "":
        result["gpt_eval_match"] = False
        result["gpt_eval_analysis"] = "No diffs to evaluate"
        return

    user_prompt = make_user_prompt(result["original_code"], diff["diff1"], diff["diff2"])

    async with semaphore:  # Limit concurrent requests
        try:
            response = await client.chat.completions.create(
                model="deepprompt-gpt-4.1-2025-04-14-global",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
            )

            analysis = response.choices[0].message.content
            gpt_eval_match = extract_result(analysis)

            result["gpt_eval_match"] = gpt_eval_match
            result["gpt_eval_analysis"] = analysis
            return

        except Exception as e:
            print(f"Error evaluating result {result['id']}: {e}")
            result["gpt_eval_match"] = False
            result["gpt_eval_analysis"] = f"Error: {str(e)}"
            return


async def evaluate_diffs_async(results: list[dict], diffs: list[dict],
                                max_concurrent: int = 10) -> list[dict]:
    """Evaluate all results asynchronously with controlled concurrency."""
    client = AsyncAzureOpenAI(
        api_version="2024-12-01-preview",
        azure_endpoint="https://deepprompteastus2.openai.azure.com/",
        api_key=os.getenv("OPENAI_API_KEY", None),
    )

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Create tasks for all evaluations
    tasks = [
        evaluate_single_result(client, semaphore, result, diffs[i])
        for i, result in enumerate(results)
    ]

    # Execute all tasks with progress bar
    for task in atqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating diff equivalences..."):
        await task

    await client.close()

    return results


async def main():
    parser = argparse.ArgumentParser(description="Grade evaluation results")
    parser.add_argument("result_file", help="Path to the results JSONL file")
    parser.add_argument("--max-concurrent", type=int, default=20,
                       help="Maximum number of concurrent API requests")

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print(f"Grading: {args.result_file}")
    print(f"{'=' * 60}")

    results = load_from_jsonl(args.result_file)
    print(f"Total test cases: {len(results)}")

    diffs = get_diffs(results)
    print(f"Total test cases after regularization: {len(diffs)}")

    # Run async evaluation
    results = await evaluate_diffs_async(results, diffs, args.max_concurrent)

    save_to_jsonl(results, args.result_file)

    # compute accuracy
    accuracy = sum(result["gpt_eval_match"] for result in results) / len(results)
    print(f"GPT-4.1 Evaluation Accuracy of {args.result_file.split('/')[-1]}: {accuracy * 100:.2f}% ({sum(result['gpt_eval_match'] for result in results)}/{len(results)})")

if __name__ == "__main__":
    asyncio.run(main())

import argparse
import asyncio
import json
import os
from typing import Any

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from utils import load_from_jsonl


class EvaluationRunner:
    def __init__(self, model: str, port: int = 30000):
        self.model = model
        self.client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    async def get_model_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get response from openai api"""
        configs = {"max_completion_tokens": 16384}
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "separate_reasoning": False
            },
            **configs,
        )
        return response.choices[0].message.content

    async def process_single_item(self, item: dict[str, Any], item_id: int) -> dict[str, Any]:
        """Process a single benchmark item and return the result."""
        system_prompt = item["system_prompt"]
        user_prompt = item["user_prompt"]
        ground_truth = item["ground_truth"]
        original_code = item["original_code"]
        language = item["metadata"]["language"]

        model_response = await self.get_model_response(system_prompt, user_prompt)

        return {
            "id": item_id,
            "language": language,
            "original_code": original_code,
            "ground_truth": ground_truth,
            "model_response": model_response,
        }

    async def run_evaluation(self, benchmark_data: list[dict[str, Any]], output_file: str, max_concurrent: int = 10):
        """Run evaluation on benchmark data with concurrency control."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(item_with_id):
            async with semaphore:
                item, item_id = item_with_id
                return await self.process_single_item(item, item_id)

        # Create tasks for all items with their IDs
        tasks = [process_with_semaphore((item, idx)) for idx, item in enumerate(benchmark_data)]

        # Process with progress bar
        results = []
        for coro in tqdm.as_completed(tasks, desc=f"Processing {len(benchmark_data)} items"):
            result = await coro
            results.append(result)

        # Sort results by ID to maintain order
        results.sort(key=lambda x: x["id"])

        # Write results to JSONL file
        with open(output_file, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        print(f"Results saved to {output_file}")


async def main():
    parser = argparse.ArgumentParser(description="Run evaluation on code editing benchmarks")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B",
        choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B"],
        help="Model to use for evaluation",
    )
    parser.add_argument("--port", type=int, default=30000, help="Port to use for evaluation")
    parser.add_argument(
        "--benchmark",
        choices=["fast_edit", "find_replace", "fully_rewrite", "all"],
        default="fast_edit",
        help="Which benchmark to run",
    )
    parser.add_argument("--max_concurrent", type=int, default=256, help="Maximum number of concurrent API calls")
    parser.add_argument("--output_dir", default="./results", help="Directory to save results")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize evaluation runner
    runner = EvaluationRunner(args.model, args.port)

    model_name = args.model.split("/")[-1]
    # Run evaluations based on benchmark choice
    if args.benchmark == "fast_edit" or args.benchmark == "all":
        print("Loading fast editing benchmark...")
        fast_edit_data = load_from_jsonl("./fast_editing_benchmark.jsonl")
        output_file = os.path.join(args.output_dir, f"{model_name}_fast_edit__results.jsonl")
        print(f"Running evaluation on fast editing benchmark with {args.model}...")
        await runner.run_evaluation(fast_edit_data, output_file, args.max_concurrent)

    if args.benchmark == "find_replace" or args.benchmark == "all":
        print("Loading find_replace benchmark...")
        find_replace_data = load_from_jsonl("./find_replace_benchmark.jsonl")
        output_file = os.path.join(args.output_dir, f"{model_name}_find_replace__results.jsonl")
        print(f"Running evaluation on find_replace benchmark with {args.model}...")
        await runner.run_evaluation(find_replace_data, output_file, args.max_concurrent)

    if args.benchmark == "fully_rewrite" or args.benchmark == "all":
        print("Loading fully_rewrite benchmark...")
        fully_rewrite_data = load_from_jsonl("./fully_rewrite_benchmark.jsonl")
        output_file = os.path.join(args.output_dir, f"{model_name}_fully_rewrite__results.jsonl")
        print(f"Running evaluation on fully_rewrite benchmark with {args.model}...")
        await runner.run_evaluation(fully_rewrite_data, output_file, args.max_concurrent)

    print("Evaluation completed!")


if __name__ == "__main__":
    asyncio.run(main())

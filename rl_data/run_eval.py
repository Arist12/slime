import argparse
import asyncio
import json
import os
from typing import Any

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI, AsyncOpenAI
from tqdm.asyncio import tqdm
from utils import load_from_jsonl

load_dotenv()

# Model deployment mappings
MODEL_DEPLOYMENTS = {
    "gpt-4o": "deepprompt-gpt-4o-2024-08-06-global",
    "gpt-4.1": "deepprompt-gpt-4.1-2025-04-14-global",
    "gpt-4.1-mini": "deepprompt-gpt-4.1-mini-2025-04-14-global",
    "gpt-5": "deepprompt-gpt-5-2025-08-07-global",
    "gpt-5-mini": "deepprompt-gpt-5-mini-2025-08-07-global",
    "gpt-5-nano": "deepprompt-gpt-5-nano-2025-08-07-global",
    "claude-4-sonnet": "claude-sonnet-4-0",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-4b-100step": "Qwen/Qwen3-4B-100step",
    "qwen-slime": "qwen-slime"
}


class EvaluationRunner:
    def __init__(self, model_name: str, port: int = 30000):
        self.client_type = "openai"  # use openai api by default
        self.model_deployment = MODEL_DEPLOYMENTS[model_name]

        if model_name.startswith("gpt"):
            self.client = AsyncAzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint="https://deepprompteastus2.openai.azure.com/",
                api_key=os.getenv("OPENAI_API_KEY", None),
            )
        elif model_name.startswith("claude"):
            self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", None))
            self.client_type = "anthropic"
        else:
            self.client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

    async def get_model_response(self, system_prompt: str, user_prompt: str) -> str:
        if self.client_type == "openai":
            return await self._get_openai_response(system_prompt, user_prompt)
        elif self.client_type == "anthropic":
            return await self._get_anthropic_response(system_prompt, user_prompt)
        else:
            raise ValueError(f"Invalid client type: {self.client_type}")

    async def _get_openai_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get response from openai api"""
        configs = {"max_completion_tokens": 16384}

        if self.model_deployment.startswith("Qwen"):  # recommended sampling parameters for Qwen3 models
            configs["temperature"] = 0.6
            configs["top_p"] = 0.95

        response = await self.client.chat.completions.create(
            model=self.model_deployment,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            **configs,
        )
        return response.choices[0].message.content

    async def _get_anthropic_response(self, system_prompt: str, user_prompt: str) -> str:
        """Get response from anthropic api"""
        response = await self.client.messages.create(
            model=self.model_deployment,
            max_tokens=16384,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

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
        choices=[
            "gpt-4o",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "claude-4-sonnet",
            "qwen3-4b",
            "qwen3-4b-100step",
            "qwen-slime",
        ],
        default="gpt-5",
        help="Model to use for evaluation",
    )
    parser.add_argument("--port", type=int, default=30000, help="Port to use for evaluation")
    parser.add_argument(
        "--benchmark",
        choices=["code_edit", "find_replace", "fully_rewrite", "both"],
        default="both",
        help="Which benchmark to run",
    )
    parser.add_argument("--max_concurrent", type=int, default=10, help="Maximum number of concurrent API calls")
    parser.add_argument("--output_dir", default="./results", help="Directory to save results")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize evaluation runner
    runner = EvaluationRunner(args.model, args.port)

    # Run evaluations based on benchmark choice
    if args.benchmark == "code_edit":
        print("Loading code_edit benchmark...")
        code_edit_data = load_from_jsonl("./code_editing_test.jsonl")
        output_file = os.path.join(args.output_dir, f"code_edit_{args.model}_results.jsonl")
        print(f"Running evaluation on code_edit benchmark with {args.model}...")
        await runner.run_evaluation(code_edit_data, output_file, args.max_concurrent)

    if args.benchmark == "find_replace" or args.benchmark == "both":
        print("Loading find_replace benchmark...")
        find_replace_data = load_from_jsonl("./find_replace_benchmark.jsonl")
        output_file = os.path.join(args.output_dir, f"find_replace_{args.model}_results.jsonl")
        print(f"Running evaluation on find_replace benchmark with {args.model}...")
        await runner.run_evaluation(find_replace_data, output_file, args.max_concurrent)

    if args.benchmark == "fully_rewrite" or args.benchmark == "both":
        print("Loading fully_rewrite benchmark...")
        fully_rewrite_data = load_from_jsonl("./fully_rewrite_benchmark.jsonl")
        output_file = os.path.join(args.output_dir, f"fully_rewrite_{args.model}_results.jsonl")
        print(f"Running evaluation on fully_rewrite benchmark with {args.model}...")
        await runner.run_evaluation(fully_rewrite_data, output_file, args.max_concurrent)

    print("Evaluation completed!")


if __name__ == "__main__":
    asyncio.run(main())

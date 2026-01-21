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


MODEL_DEPLOYMENTS = {
    "gpt-41": "deepprompt-gpt-4.1-2025-04-14-global",
    "gpt-5": "gpt-5",
    "gpt-5-mini": "deepprompt-gpt-5-mini-2025-08-07-global",
    "gpt-5-nano": "deepprompt-gpt-5-nano-2025-08-07-global",
    "claude-sonnet-45": "claude-sonnet-4-5",
    "claude-haiku-45": "claude-haiku-4-5"
}

BENCHMARKS_PATHS = {
    "adpative": "./benchmarks/llm_code_editing_test.jsonl",
    "find_replace": "./benchmarks/find_replace_test.jsonl",
    "full_rewrite": "./benchmarks/full_rewrite_test.jsonl",
}




class EvaluationRunner:
    def __init__(self, model: str, port: int = 30000):
        self.model = model
        self.client_type = "openai" if not "claude" in self.model.lower() else "anthropic"

        self._set_client(self.model, port)

    def _set_client(self, model: str, port: int):
        if "claude" in model.lower():
            self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", None))
        elif model in MODEL_DEPLOYMENTS:
            self.client = AsyncAzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint="https://deepprompteastus2.openai.azure.com/",
                api_key=os.getenv("OPENAI_API_KEY", None),
            )
        else:
            self.client = AsyncOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

        if model in MODEL_DEPLOYMENTS:
            self.model = MODEL_DEPLOYMENTS[model]

    async def get_model_response(self, user_prompt: str) -> str:
        if self.client_type == "openai":
            return await self._get_openai_response(user_prompt)
        elif self.client_type == "anthropic":
            return await self._get_anthropic_response(user_prompt)
        else:
            raise ValueError(f"Invalid client type: {self.client_type}")

    async def _get_openai_response(self, user_prompt: str) -> str:
        """Get response from openai api"""
        messages = [{"role": "user", "content": user_prompt}]

        if "qwen" in self.model.lower():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=32768,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "separate_reasoning": False
                }
            )
        else:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=32768
            )
        return response.choices[0].message.content

    async def _get_anthropic_response(self, user_prompt: str) -> str:
        """Get response from anthropic api"""
        response_text = ""
        async with self.client.messages.stream(
            model=self.model,
            max_tokens=32768,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            async for text in stream.text_stream:
                response_text += text
        return response_text

    async def process_single_item(self, item: dict[str, Any], item_id: int) -> dict[str, Any]:
        """Process a single benchmark item and return the result."""
        user_prompt = item["prompt"][0]["content"]
        ground_truth = item["label"]
        original_code = item["metadata"]["original_code"]
        language = item["metadata"]["langauge"]

        model_response = await self.get_model_response(user_prompt)

        return {
            "id": item_id,
            "prompt": user_prompt,
            "original_code": original_code,
            "ground_truth": ground_truth,
            "model_response": model_response,
            "language": language,
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
        default="Qwen/Qwen3-8B",
        choices=["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]+list(MODEL_DEPLOYMENTS.keys()),
        help="Model to use for evaluation",
    )
    parser.add_argument("--port", type=int, default=30000, help="Port to use for evaluation")
    parser.add_argument("--max_concurrent", type=int, default=500, help="Maximum number of concurrent API calls")
    parser.add_argument("--tag", type=str, default="qwen", help="Tag to include in the output file name")
    parser.add_argument("--benchmark", type=str, default="adpative", choices=list(BENCHMARKS_PATHS.keys()), help="Benchmark to use for evaluation")
    parser.add_argument("--output_dir", default="./results", help="Directory to save results")

    args = parser.parse_args()

    # Sample Run:
    # python run_eval.py --tag qwen3-8b-rl --benchmark adpative

    async def load_and_run_benchmark(tag: str):
        print(f"Loading  benchmark...")
        benchmark_data = load_from_jsonl(BENCHMARKS_PATHS[args.benchmark])
        output_file = os.path.join(args.output_dir, f"{tag}_{args.benchmark}_results.jsonl")
        print(f"Running evaluation on {args.benchmark} benchmark with {args.model}...")
        await runner.run_evaluation(benchmark_data, output_file, args.max_concurrent)

    os.makedirs(args.output_dir, exist_ok=True)
    runner = EvaluationRunner(args.model, args.port)
    await load_and_run_benchmark(args.tag)

    print("Evaluation completed!")


if __name__ == "__main__":
    asyncio.run(main())

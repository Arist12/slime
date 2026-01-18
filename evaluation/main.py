import json

with open("./benchmarks/llm_code_editing_test.jsonl", "r") as f:
    data = [json.loads(line) for line in f]

print(data[0].keys())
print(data[0]["prompt"][0].keys())
print(data[0]["metadata"].keys())
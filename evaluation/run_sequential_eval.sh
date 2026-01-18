python run_gpt_grader.py /mnt/local/yikai/slime/evaluation/results/gpt-5-0_new_results_graded.jsonl
python run_gpt_grader.py /mnt/local/yikai/slime/evaluation/results/gpt-5-1_new_results_graded.jsonl
python run_gpt_grader.py /mnt/local/yikai/slime/evaluation/results/gpt-5-2_new_results_graded.jsonl
python run_gpt_grader.py /mnt/local/yikai/slime/evaluation/results/gpt-5-mini-1_new_results_graded.jsonl
python run_gpt_grader.py /mnt/local/yikai/slime/evaluation/results/gpt-5-mini-2_new_results_graded.jsonl
python run_gpt_grader.py /mnt/local/yikai/slime/evaluation/results/gpt-5-mini-3_new_results_graded.jsonl
python run_gpt_grader.py /mnt/local/yikai/slime/evaluation/results/claude-sonnet-45-0_new_results_graded.jsonl

python run_eval.py --model claude-haiku-45 --tag claude-haiku-45-0 --max_concurrent 10
python run_eval.py --model claude-sonnet-45 --tag claude-sonnet-45-0 --max_concurrent 10

# Step 0: Install the humaneval benchmark
git clone https://github.com/openai/human-eval
pip install -e human-eval

# Step 1: Run the inference
python -m benchmarks.humaneval.inference \
  --data_name openai/openai_humaneval \
  --split test \
  --output_file benchmarks/humaneval/llm_eval_prediction.jsonl \
  --on_aios \
  --agent_type llm

# Step 2: Evaluate the functional correctness
evaluate_functional_correctness benchmarks/humaneval/llm_eval_prediction.jsonl
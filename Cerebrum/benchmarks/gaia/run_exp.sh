# Step 1: Run the inference

python -m benchmarks.gaia.inference \
  --data_name gaia-benchmark/GAIA \
  --split validation \
  --output_file benchmarks/gaia/react_eval_prediction.json \
  --on_aios \
  --agent_type react

python -m benchmarks.agents.react \
  --data_name gaia-benchmark/GAIA \
  --split validation \
  --output_file benchmarks/gaia/react_eval_prediction.json \
  --on_aios \
  --agent_type react

# Step 2: Run the evaluation script
# python -m benchmarks.gaia.inference \
#   --data_name gaia-benchmark/GAIA \
#   --split validation \
#   --output_file benchmarks/gaia/llm_eval_prediction.json \
#   --on_aios \
#   --agent_type llm
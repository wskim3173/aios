# Step 0: Install the SWE-bench
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench
pip install -e .

# Step 0.1: Run the test code to check if the installation is successful
python -m swebench.harness.run_evaluation \
    --predictions_path gold \
    --max_workers 1 \
    --instance_ids sympy__sympy-20590 \
    --run_id validate-gold

# Step 1: Run the inference
python -m benchmarks.swebench.inference \
  --data_name princeton-nlp/SWE-bench_Lite \
  --split test \
  --output_file benchmarks/swebench/eval_prediction.json \
  --on_aios \
  --agent_type llm

# Step 2: Evaluate the functional correctness

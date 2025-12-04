1. venv 활성화
    source .venv311/bin/activate

2. AIOS 실행
    bash runtime/launch_kernal.sh

3. Llama 서버 실행
   vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8091 \
  --max-model-len 2048 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.88 \
  --dtype bfloat16

4. HumanEval 평가 실행
   python -m benchmarks.humaneval.inference_multi \
  --data_name openai/openai_humaneval \
  --split test \
  --output_file humaneval/output_multi.jsonl \
  --agent_type react \
  --on_aios \
  --agent_num 64 \
  --max_num 164

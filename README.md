1. venv 활성화
    source .venv311/bin/activate

2. AIOS 실행
    bash runtime/launch_kernel.sh

3. Llama 서버 실행
   vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8091 \
  --max-model-len 3072 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.88 \
  --dtype bfloat16

4. HumanEval 평가 실행
   python -m benchmarks.humaneval.inference_multi \
  --data_name openai/openai_humaneval \
  --split test \
  --output_file humaneval/output_multi.jsonl \
  --agent_type react_code \
  --on_aios \
  --agent_num 32 \
  --max_num 164

5. 채점
   evaluate_functional_correctness humaneval/output_multi.jsonl

6. 참고
   aios 없이 gpt를 사용할때는 OPEN_API_KEY를 미리 export 해둬야 한다
   export OPENAI_API_KEY="sk-proj-..."

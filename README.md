1. AIOS 설치
  git clone https://github.com/wskim3173/aios.git
  cd AIOS
  sudo apt update
  python3.11 -m venv venv311
  source .venv311/bin/activate
  pip install uv
  uv pip install -r requirements-cuda.txt

2. Cerebrum 설치
  cd Cerebrum && uv pip install -e .
  cd ..

3. HumanEval 설치
  cd ./human-eval/
  pip install -e .

4. vLLM 설치
  pip install vllm --upgrade
  sudo apt install python3.11-dev
  ※ llama 다운받으려면 권한이 필요합니다. 아래 링크 참고 해주세요
  https://docs.google.com/presentation/d/1JLO_MnYwe7esGA4ps3zP2wOSx5FqYDsDOfsxAJOSg7Q/edit?slide=id.g3a479621cf4_128_30#slide=id.g3a479621cf4_128_30

5. 관련 패키지 최신화
  pip install --upgrade datasets
  pip install --upgrade fsspec
  pip install --upgrade sentence-transformers

6. Config 수정
  openai, huggingface에 토큰 입력

7. AIOS 실행
  bash runtime/launch_kernel.sh

8. (다른 터미널 열고) Llama 서버 실행
  source .venv311/bin/activate

  vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8091 \
  --max-model-len 3072 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.88 \
  --dtype bfloat16

9. (다른 터미널 열고) HumanEval 평가 실행
  source .venv311/bin/activate

  cd ./AIOS/Cerebrum/benchmarks/

  python -m benchmarks.humaneval.inference_multi \
  --data_name openai/openai_humaneval \
  --split test \
  --output_file humaneval/output_multi.jsonl \
  --agent_type react_code \
  --on_aios \
  --agent_num 32 \
  --max_num 164

10. 채점
  evaluate_functional_correctness humaneval/output_multi.jsonl

11. 참고
   aios 없이 gpt를 사용할때는 OPEN_API_KEY를 미리 export 해둬야 한다
   export OPENAI_API_KEY="sk-proj-..."

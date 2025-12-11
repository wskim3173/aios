#!/usr/bin/env python3
# benchmarks/humaneval/inference.py

import json
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from tqdm import tqdm

# 그대로 재사용: AGENT 매핑과 파이프라인 메타
from ..experiment_core import MetaData, AGENT_TYPE_MAPPING_AIOS  # :contentReference[oaicite:2]{index=2}
from ..utils import get_parser  # :contentReference[oaicite:3]{index=3}

# ===== helpers (기존 inference.py의 로직을 유지) =====
def parse_result(result: str) -> str:
    start_idx = result.find("<FINAL_ANSWER>")
    end_idx = result.find("</FINAL_ANSWER>")
    if start_idx != -1 and end_idx != -1:
        return result[start_idx + len("<FINAL_ANSWER>"):end_idx]
    return ""

def write_output_func(result_list: List[Dict[str, Any]], output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        for r in result_list:
            f.write(json.dumps(r) + "\n")
    print(f"Write results num: {len(result_list)}")

def build_check_program(prompt: str, completion: str, test: str, entry_point: str) -> str:
    return (
        prompt + completion + "\n" + test + "\n" + f"check({entry_point})"
    )

# ===== core =====
def process_one_with_agent(agent, data: Dict[str, Any], programs_dir: str) -> Dict[str, Any]:
    # agent.run_humaneval은 기존 구현과 동일한 인터페이스를 기대
    result = agent.run(data["prompt"])
    #result = parse_result(result)

    check_program = build_check_program(
        prompt=data["prompt"],
        completion=result,
        test=data["test"],
        entry_point=data["entry_point"],
    )
    task_id = data["task_id"].split("/")[-1]

    os.makedirs(programs_dir, exist_ok=True)
    with open(os.path.join(programs_dir, f"program{task_id}.py"), "w", encoding="utf-8") as f:
        f.write(check_program)

    return {"task_id": data["task_id"], "completion": result}

def main():
    # 기본 파서는 재사용하되 멀티에이전트 옵션 추가
    parser = get_parser()  # --data_name, --split, --output_file, --on_aios, --max_num 등
    #parser.add_argument("--agent_type", type=str, default="react",
    #                    help="Agent kind for HumanEval (e.g., react, cot)")
    parser.add_argument("--agent_num", type=int, default=1,
                        help="Number of concurrent agents (threads)")
    args = parser.parse_args()

    # agent_type 해석: experiment_core의 키 규칙을 그대로 사용
    full_agent_key = f"humaneval:{args.agent_type}"
    if full_agent_key not in AGENT_TYPE_MAPPING_AIOS:
        valid = ", ".join(sorted(k.split(":")[1] for k in AGENT_TYPE_MAPPING_AIOS if k.startswith("humaneval:")))
        raise ValueError(f"Unknown agent_type '{args.agent_type}'. Valid for HumanEval: {valid}")

    AgentClass = AGENT_TYPE_MAPPING_AIOS[full_agent_key]

    # 데이터셋 로드
    dataset = load_dataset(args.data_name, split=args.split)

    # 실행 상한
    max_num = args.max_num if args.max_num is not None else len(dataset)
    items = list(dataset)[:max_num]

    # 에이전트 풀 생성 (동시성은 스레드풀로)
    agents = [AgentClass(args.on_aios) for _ in range(max(1, args.agent_num))]

    # 프로그램 저장 경로 (기존 경로 스타일 유지)
    programs_dir = os.path.join(os.path.dirname(__file__), "programs")

    # 제출 작업 생성
    results: List[Dict[str, Any]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max(1, args.agent_num)) as ex:
        futures = {}
        for idx, data in enumerate(items):
            agent = agents[idx % len(agents)]
            fut = ex.submit(process_one_with_agent, agent, data, programs_dir)
            futures[fut] = idx

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Running HumanEval (multi-agent)"):
            idx = futures[fut]
            results[idx] = fut.result()

    # 출력 저장
    write_output_func(results, args.output_file)

if __name__ == "__main__":
    main()

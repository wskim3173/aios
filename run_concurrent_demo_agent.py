#!/usr/bin/env python3
# run_concurrent_demo_agents.py
#
# - Local agent 목록에서 'demo_agent' 디렉터리를 찾아 실행
# - 각 실행은 고유 임시 디렉터리(TMPDIR)로만 격리 (XDG_CACHE_HOME 공용)
# - 시작 전에 툴 프리로드
# - Warm-up 1회 후 동시 실행
# - 결과는 JSONL 저장

import os
import json
import time
import argparse
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List
from uuid import uuid4
import traceback

from cerebrum.manager.agent import AgentManager
from cerebrum.config.config_manager import config
from cerebrum.commands.run_agent import AgentConfig, AgentRunner
from cerebrum.interface import AutoTool

DEFAULT_NUM_AGENTS = 8
SUBMIT_CONCURRENCY = 8
GLOBAL_CACHE = os.environ.get("XDG_CACHE_HOME", str(Path.home()/".cache"))
GLOBAL_CEREBRUM_CACHE = str(Path(GLOBAL_CACHE) / "cerebrum")  # 공용
TASK_TEMPLATE = "Give me a two-sentence summary about AI scheduling and queuing. Req#{i}"

def preload_tools():
    """동시 실행 전 arxiv 툴을 한 번만 받아 캐시에 준비."""
    try:
        AutoTool.from_preloaded("demo_author/arxiv")
        print("[INFO] Preloaded tool: demo_author/arxiv")
    except Exception:
        try:
            AutoTool.from_preloaded("arxiv")
            print("[INFO] Preloaded tool: arxiv")
        except Exception as e:
            print(f"[WARN] Tool preload failed: {e!r}. Will rely on warm-up run.")

def make_tasks(n: int) -> List[str]:
    return [TASK_TEMPLATE.format(i=i) for i in range(n)]

def _with_env(env_updates: dict, fn, *args, **kwargs):
    """Temporarily apply env vars, run fn, then restore."""
    old = {}
    to_del = []
    try:
        for k, v in env_updates.items():
            if k in os.environ:
                old[k] = os.environ[k]
            else:
                to_del.append(k)
            os.environ[k] = v
        return fn(*args, **kwargs)
    finally:
        for k, v in old.items():
            os.environ[k] = v
        for k in to_del:
            os.environ.pop(k, None)

def make_run_env(run_id: str) -> dict:
    run_root = Path(tempfile.gettempdir()) / f"demo_agent_run_{run_id}"
    (run_root / "tmp").mkdir(parents=True, exist_ok=True)
    # XDG_CACHE_HOME 은 건드리지 않음 (공용 캐시 사용)
    return {
        "TMPDIR": str(run_root / "tmp"),
        # 필요 시 HTTPX 캐시를 격리하려면 아래를 사용 (이슈 있으면 주석 처리)
        # "HTTPX_CACHE_DIR": str(run_root / "tmp"),
    }

def copy_agent_dir(src: Path) -> Path:
    """Per-run copy of the agent folder to avoid shared writes/read races."""
    run_id = uuid4().hex[:8]
    dst = Path(tempfile.gettempdir()) / f"demo_agent_dir_{run_id}"
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst)
    return dst

def _extract_text(result_obj) -> str:
    # demo_agent는 {"result": "..."} 형태가 일반적이나, 혹시 모를 변형에 대비
    if isinstance(result_obj, str):
        return result_obj
    if not isinstance(result_obj, dict):
        return ""
    for k in ("result", "final", "text", "message", "output"):
        v = result_obj.get(k)
        if isinstance(v, str) and v.strip():
            return v
    data = result_obj.get("data") or result_obj.get("payload")
    if isinstance(data, dict):
        for k in ("result", "final", "text", "message", "output"):
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v
    return ""

async def run_one_agent(agent_dir: Path, task: str, agenthub_url: str) -> Dict[str, Any]:
    unique = uuid4().hex[:8]

    # per-run temp isolation
    env_updates = make_run_env(unique)

    cfg = AgentConfig(
        agent_path=str(agent_dir),            # directory, not zip
        agent_name=f"demo_agent_{unique}",    # unique name
        task_input=task,
        agenthub_url=agenthub_url,
        mode="local",
    )
    runner = AgentRunner(cfg)

    t0 = time.time()
    attempts = 0
    result_obj = None
    try:
        # run AgentRunner in a thread, with env isolation inside the thread
        def _call():
            return _with_env(env_updates, runner.run)
        result_obj = await asyncio.to_thread(_call)
    except SystemExit:
        # transient, retry once
        attempts += 1
        await asyncio.sleep(0.4)
        result_obj = await asyncio.to_thread(_call)
    except Exception:
        # return rich error
        elapsed = round(time.time() - t0, 2)
        return {
            "ok": False,
            "elapsed_sec": elapsed,
            "error": "Exception during AgentRunner.run()",
            "trace": traceback.format_exc()[:4000],
        }

    elapsed = round(time.time() - t0, 2)
    text = _extract_text(result_obj).strip()
    ok = bool(text)

    return {
        "ok": ok,
        "elapsed_sec": elapsed,
        "attempts": attempts + 1,
        "result": (text[:2000] + ("..." if len(text) > 2000 else "")),
    }

async def main(num_agents: int):
    agent_hub_url = config.get_agent_hub_url()
    am = AgentManager(agent_hub_url)
    local_agents = am.list_local_agents()
    if not local_agents:
        raise RuntimeError("No local agents found.")

    # Pick demo_agent directory
    demo = next((a for a in local_agents if (a.get("name") or "").lower() == "demo_agent"), None)
    if demo is None:
        raise RuntimeError("demo_agent not found among local agents.")

    demo_dir = Path(demo.get("path", ""))
    if not demo_dir or not demo_dir.is_dir():
        raise RuntimeError(f"demo_agent path is invalid: {demo_dir}")

    print(f"[INFO] Using demo_agent at: {demo_dir}")

    # Warm-up (build caches/deps once)
    print("[INFO] Warm-up run...")
    _ = await run_one_agent(demo_dir, "WARMUP: say hi", agent_hub_url)
    print("[INFO] Warm-up done. Launching concurrent runs...")

    tasks_payloads = make_tasks(num_agents)

    sem = asyncio.Semaphore(SUBMIT_CONCURRENCY)

    async def guarded_run(task_str: str):
        # Per-run copy of agent dir
        agent_copy = copy_agent_dir(demo_dir)
        async with sem:
            try:
                return await run_one_agent(agent_copy, task_str, agent_hub_url)
            finally:
                shutil.rmtree(agent_copy, ignore_errors=True)

    results = await asyncio.gather(*(guarded_run(t) for t in tasks_payloads))

    ok_cnt = sum(1 for r in results if r.get("ok"))
    print(f"[INFO] Completed {len(results)} runs | success={ok_cnt} | fail={len(results)-ok_cnt}")

    out_dir = Path("test_results")
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"demo_agent_concurrent_{int(time.time())}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[INFO] Results saved to {out_file.resolve()}")

if __name__ == "__main__":
    # 동시 실행 전에 툴 프리로드 (공용 캐시 사용)
    preload_tools()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_agents", type=int, default=DEFAULT_NUM_AGENTS,
                        help="Number of concurrent demo_agent runs.")
    args = parser.parse_args()
    asyncio.run(main(args.num_agents))

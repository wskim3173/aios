# benchmarks/agents/react.py
from typing import Dict, Any, List, Optional
import time

from cerebrum.llm.apis import llm_chat_with_json_output
from cerebrum.utils import _parse_json_output


class ReActAgent:
    """
    Minimal, tool-free ReAct agent.
    - No external agents, no browser, no MCP, no code executor.
    - Pure LLM self-refine loop (reason-only). 
    - For HumanEval: returns ONLY the function body text.

    Loop:
      Reason  : propose/critique plan for implementation
      (Act)   : internally revise the code draft (no external call)
      Observe : reflect on issues and decide to finish or continue
    """

    def __init__(self, on_aios: bool = True, max_steps: int = 4):
        self.agent_name = "react"
        self.on_aios = on_aios
        self.max_steps = max_steps
        self.history: List[Dict[str, Any]] = []
        # LLM 라우팅(AIOS가 백엔드 선택). 필요시 모델 변경 가능.
        self.llms = [{"name": "gpt-4o-mini", "backend": "openai"}]

    # ---- Public API ----
    def run_humaneval(self, prompt: str) -> str:
        """
        HumanEval 한 항목에 대해 최종 '함수 본문'만 반환.
        """
        code = self._react_reason_only(prompt)
        return self._strip_code_fences(code).strip()

    def run(self, task_input: str) -> Dict[str, Any]:
        """
        일반 태스크용. 최종 산출(코드나 답변)을 'final'에 담아 반환.
        """
        text = self._react_reason_only(task_input)
        return {"agent_name": self.agent_name, "final": text}

    # ---- Internal ----
    def _react_reason_only(self, task_input: str) -> str:
        """
        외부 툴/에이전트 없이 LLM 자기점검 루프만으로 개선.
        JSON I/O로 일관성 유지.
        """
        self.history.clear()

        system_prompt = f"""You are a senior Python assistant solving a code-completion task.
Follow a compact ReAct loop (Reason → Revise → Judge). 
No external tools. You must end with a clean function BODY only (no def header if it is already present in prompt).

## Task
{task_input}

## Rules
- Think concisely about pitfalls (edge cases, complexity, off-by-one, empty inputs).
- Produce or refine a candidate implementation (function BODY only).
- Never include explanations in the final body.
- When you are fully confident, set finish=true and stop.

## JSON response schema (STRICT)
{{
  "observation": "what the previous draft did well/poorly (short)",
  "reasoning":   "what to change or verify (short)",
  "finish":      true or false,
  "candidate":   "FUNCTION BODY ONLY (no ``` fences, no def line unless the prompt asks for body only inside the function)"
}}
"""

        messages = [{"role": "system", "content": system_prompt}]
        final_code = ""

        for step in range(self.max_steps):
            step_prompt = f"""History (latest first, up to 3):
{self.history[-3:]}
Return STRICT JSON with keys: observation, reasoning, finish, candidate."""
            messages.append({"role": "user", "content": step_prompt})

            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "react_reason_only",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "observation": {"type": "string"},
                            "reasoning":   {"type": "string"},
                            "finish":      {"type": "boolean"},
                            "candidate":   {"type": "string"}
                        },
                        "required": ["observation", "reasoning", "finish", "candidate"]
                    },
                    "strict": True
                }
            }

            resp = llm_chat_with_json_output(
                agent_name=self.agent_name,
                messages=messages,
                llms=self.llms,
                response_format=response_format
            )

            raw = resp["response"]["response_message"]
            step_obj = _parse_json_output(raw) or {}

            obs = (step_obj.get("observation") or "").strip()
            rsn = (step_obj.get("reasoning") or "").strip()
            fin = bool(step_obj.get("finish", False))
            cand = (step_obj.get("candidate") or "").strip()

            self.history.append({
                "round": step,
                "observation": obs,
                "reasoning": rsn,
                "finish": fin,
                "candidate_len": len(cand)
            })

            # 최신 후보를 보정/보관
            final_code = self._strip_code_fences(cand).strip()

            if fin and final_code:
                break

            time.sleep(0.1)

        # 최종 산출을 FINAL_ANSWER태그 형식으로 감싸 inference.py와 궁합 유지(필요시)
        return f"<FINAL_ANSWER>\n{final_code}\n</FINAL_ANSWER>"

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """
        ``` 또는 ```python 코드펜스 제거.
        """
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            if t.startswith("python"):
                t = t[len("python"):].lstrip()
        return t

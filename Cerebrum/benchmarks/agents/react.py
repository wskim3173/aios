from typing import Dict, Any, List
import time

from cerebrum.llm.apis import llm_chat_with_json_output
from cerebrum.utils import _parse_json_output


class ReActAgent:
    """
    ReActAgent (AIOS-ON version)
    - Uses llm_chat_with_json_output (AIOS backend)
    - No external tools, pure reasoning loop
    - Returns properly indented function body for HumanEval
    """

    def __init__(self, on_aios: bool = True, max_steps: int = 4):
        self.agent_name = "react"
        self.on_aios = on_aios  # 기본: True
        self.max_steps = max_steps
        self.history: List[Dict[str, Any]] = []
        self.llms = [{"name": "gpt-4o-mini", "backend": "openai"}]

    # --- Core Loop ---
    def run_humaneval(self, task_input: str) -> str:
        """
        ReAct 루프: Reason → Revise → Judge (AIOS 기반 JSON 루프)
        """
        self.history.clear()

        system_prompt = f"""You are a senior Python assistant solving a code-completion task.
Follow a compact ReAct loop (Reason → Revise → Judge).
No external tools. You must end with a clean function BODY only (no def header).

## Task
{task_input}

## Rules
- Think concisely about pitfalls (edge cases, off-by-one, etc.)
- Produce or refine a candidate implementation (function BODY only)
- Never include explanations in the final body
- When confident, set finish=true and stop.

## JSON response schema (STRICT)
{{
  "observation": "what the previous draft did well/poorly (short)",
  "reasoning":   "what to change or verify (short)",
  "finish":      true or false,
  "candidate":   "FUNCTION BODY ONLY (no def line, no code fences)"
}}
"""

        messages = [{"role": "system", "content": system_prompt}]
        final_code = ""

        # --- JSON Schema (OpenAI-compatible) ---
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "react_reason_only",
                "schema": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "observation": {"type": "string"},
                        "reasoning":   {"type": "string"},
                        "finish":      {"type": "boolean"},
                        "candidate":   {"type": "string"},
                    },
                    "required": ["observation", "reasoning", "finish", "candidate"]
                },
                "strict": True
            }
        }

        for step in range(self.max_steps):
            step_prompt = f"""History (latest first, up to 3):
{self.history[-3:]}
Return STRICT JSON with keys: observation, reasoning, finish, candidate."""
            messages.append({"role": "user", "content": step_prompt})

            # --- AIOS 통한 JSON 응답 ---
            resp = llm_chat_with_json_output(
                agent_name=self.agent_name,
                messages=messages,
                llms=self.llms,
                response_format=response_format
            )
            raw = resp["response"]["response_message"]

            step_obj = _parse_json_output(raw) or {}

            obs = step_obj.get("observation").strip()
            rsn = step_obj.get("reasoning").strip()
            fin = step_obj.get("finish", False)
            cand = step_obj.get("candidate").strip()

            self.history.append({
                "round": step,
                "observation": obs,
                "reasoning": rsn,
                "finish": fin,
                "candidate_len": len(cand)
            })

            final_code = self._strip_code_fences(cand).strip()

            if fin and final_code:
                break

            time.sleep(0.1)
        
        final_code = self._normalize_body_indent(final_code)
        return f"<FINAL_ANSWER>\n{final_code}\n</FINAL_ANSWER>"

    # --- Helper methods ---
    @staticmethod
    def _strip_code_fences(text: str) -> str:
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            if t.startswith("python"):
                t = t[len("python"):].lstrip()
        return t

    @staticmethod
    def _normalize_body_indent(code: str, spaces: int = 4) -> str:
        """
        HumanEval용 함수 본문 들여쓰기 정규화:
        - 앞뒤 빈 줄 제거
        - 공통 최소 들여쓰기(dedent) 제거
        - 정확히 `spaces`칸 들여쓰기 적용
        - 탭을 스페이스로 치환
        """
        if code is None:
            return "\n"

        # 줄 단위 분리
        lines = code.splitlines()

        # 앞/뒤 공백 줄 제거
        while lines and lines[0].strip() == "":
            lines.pop(0)
        while lines and lines[-1].strip() == "":
            lines.pop()

        if not lines:
            return "\n"

        # 탭을 스페이스로 치환
        lines = [ln.replace("\t", "    ") for ln in lines]

        # 공통 최소 선행 공백 폭 계산
        def leading_spaces(s: str) -> int:
            return len(s) - len(s.lstrip(" "))

        min_lead = None
        for ln in lines:
            if ln.strip() == "":
                continue
            lead = leading_spaces(ln)
            min_lead = lead if min_lead is None else min(min_lead, lead)

        if min_lead is None:
            min_lead = 0

        # dedent 적용
        dedented = []
        for ln in lines:
            if ln.strip() == "":
                dedented.append("")  # 빈 줄은 그대로 유지
            else:
                # 선행 공백이 min_lead보다 적게 들어왔을 수 있으니 방어적으로 처리
                cur_lead = leading_spaces(ln)
                cut = min(cur_lead, min_lead)
                dedented.append(ln[cut:])

        # 정확히 `spaces`칸 들여쓰기 재적용 + \n 유지
        prefix = " " * spaces
        return "".join(prefix + ln + "\n" for ln in dedented)
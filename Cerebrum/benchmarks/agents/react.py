from typing import Dict, Any, List
import time
import re
import threading

from cerebrum.llm.apis import llm_chat   # ✅ JSON이 아니라 일반 채팅 API 사용
from cerebrum.config.config_manager import config
from litellm import completion

aios_kernel_url = config.get_kernel_url()

class ReActAgent:
    """
    ReActAgent (plain-text version)
    - No JSON schema
    - No external tools
    - Simple string template + regex parsing
    - Returns properly indented function body for HumanEval
    """

    def __init__(self, on_aios: bool = True, max_steps: int = 4):
        self.agent_name = "react"
        self.on_aios = on_aios
        self.max_steps = max_steps
        self.model = "meta-llama/Llama-3.1-8B-Instruct"   #gpt-4o-mini #qwen3:1.7b #meta-llama/Llama-3.1-8B-Instruct
        self.backend = "vllm"     #openai #ollama #vllm
        self.history: List[Dict[str, Any]] = []
        self.llms = [{"name": self.model, "backend": self.backend}]
        self.t = threading.current_thread()

    # --- Core Loop ---
    def run(self, task_input: str) -> str:
        self.history.clear()

        if self.model == "meta-llama/Llama-3.1-8B-Instruct":
           system_prompt = f"""
You are a senior Python assistant solving a code-completion task.
Follow a compact ReAct loop (Reason → Revise → Judge).
No external tools.

# CONTEXT
You are completing ONLY the inside of a Python function.
The function header (def ...) already exists.
You must write only the function BODY.

# OUTPUT CONTRACT (STRICT)
Output EXACTLY in the following structure:

Thought: <one short sentence about what to fix/check>
Candidate:
<FINAL_ANSWER>
    [your Python code here, each line indented with EXACTLY 4 spaces]
</FINAL_ANSWER>
Finish: <true|false>

# RULES
1. Do NOT include 'def', parameters, or docstring.
2. Each code line MUST begin with **exactly 4 spaces**.
3. The first non-empty line must be "<FINAL_ANSWER>".
4. The last non-empty line must be "</FINAL_ANSWER>".
5. No markdown fences, no extra explanations.
6. If you output code without indentation, it is INVALID.

# EXAMPLE
Thought: Need to sort the list and compare adjacent elements.
Candidate:
<FINAL_ANSWER>
    numbers.sort()
    for i in range(len(numbers)-1):
        if numbers[i+1] - numbers[i] < threshold:
            return True
    return False
</FINAL_ANSWER>
Finish: true

## Task
{task_input}
"""
        else:
            system_prompt = f"""You are a senior Python assistant solving a code-completion task.
Follow a compact ReAct loop (Reason → Revise → Judge).
No external tools. You must end with a clean FUNCTION BODY only (no def header).

## Task
{task_input}

## Output format (STRICT - plain text, no markdown fences):
Thought: <one short sentence about what to fix/check>

Candidate:
The Candidate should strictly follow the following format and requirements:

    Format:
    Print ONLY the following wrapper with your code body inside.
    The FIRST non-empty line MUST be exactly "<FINAL_ANSWER>".
    The LAST non-empty line MUST be exactly "</FINAL_ANSWER>".
    No markdown fences. No comments. No explanations.

    Example:
    <FINAL_ANSWER>
        result = x * 2
        return result
    </FINAL_ANSWER>

    Requirements: 
    1. YOUR FINAL ANSWER must be a piece of code that can be directly filled into the given code at the <CURRENT_CURSOR_POSITION> marker.
    2. Only include the code you're adding, don't include the original function definition or comments.
    3. Do not use extra code quotes like ```python``` to wrap the code.
    4. Make sure the syntax of the code is correct, especially pay attention to proper indentation.
    5. Maintain the same indentation level as the surrounding code.
    6. If you're completing a function body, ensure all code is properly indented inside the function.
    7. Check that all return statements, loops, and conditional blocks have correct indentation.
    8. Ensure your code aligns with the original code style and indentation pattern.

Finish: <true|false>

Rules:
- Think briefly about pitfalls (edge cases, off-by-one, etc.)
- If confident the candidate is final, set Finish: true
- Otherwise, set Finish: false and wait for the next round
- Output NOTHING except the three fields above
"""

        messages = [{"role": "system", "content": system_prompt}]
        final_code = ""

        for step in range(self.max_steps):
            step_prompt = f"""History (latest first, up to 3):
{self._format_history_for_prompt(self.history[-3:])}

Now produce exactly:
Thought: ...
Candidate:
...
Finish: <true|false>"""
            messages.append({"role": "user", "content": step_prompt})

            # --- Branch on AIOS / non-AIOS ---
            if self.on_aios:
                resp = llm_chat(
                    agent_name=self.agent_name,
                    messages=messages,
                    base_url=aios_kernel_url,
                    llms=self.llms,
                )
                raw = resp["response"]["response_message"]
            else:
                if self.model == "meta-llama/Llama-3.1-8B-Instruct": 
                    non_aios_resp = completion(
                        model="hosted_vllm/"+self.model,
                        messages=messages,
                        base_url="http://127.0.0.1:8091/v1",
                        temperature=0.2,
                    )
                else:
                    non_aios_resp = completion(
                        model=self.model,
                        messages=messages,
                        temperature=0.2,
                    )

                if isinstance(non_aios_resp, str):
                    raw = non_aios_resp
                else:
                    raw = (
                        non_aios_resp.get("choices", [{}])[0]
                                     .get("message", {})
                                     .get("content")
                    )
                    if raw is None:
                        raw = non_aios_resp.get("content", "")
                raw = raw or ""

            # --- Plain ReAct parsing ---
            step_obj = self._parse_plain_react(raw)
            thought = step_obj["thought"]
            cand    = step_obj["candidate"]
            finish  = step_obj["finish"]

            self.history.append({
                "round": step,
                "thought": thought,
                "candidate_len": len(cand),
                "finish": finish,
            })

            final_code = cand

            if finish and final_code:
                break

            time.sleep(0.1)

        return final_code

    # --- Helper: format history into short lines ---
    @staticmethod
    def _format_history_for_prompt(hist: List[Dict[str, Any]]) -> str:
        if not hist:
            return "[]"
        lines = []
        for h in reversed(hist):
            lines.append(f"- Round {h['round']}: thought_len={len(h.get('thought',''))}, candidate_len={h.get('candidate_len',0)}, finish={h.get('finish',False)}")
        return "\n".join(lines)

    # --- Helper: plain-text ReAct parser ---
    @staticmethod
    def _parse_plain_react(text: str) -> Dict[str, Any]:
        """
        Expecting exactly:
        Thought: ...
        Candidate:
        ...
        Finish: true/false
        """
        # normalize line endings
        t = text.replace("\r\n", "\n").replace("\r", "\n")

        # 1) Thought
        m_thought = re.search(r"(?im)^\s*Thought:\s*(.*)$", t)
        thought = (m_thought.group(1).strip() if m_thought else "")

        # 2) Candidate block: from 'Candidate:' line until the next 'Finish:' line (or end)
        m_cand_start = re.search(r"(?im)^\s*Candidate:\s*$", t)
        candidate = ""
        if m_cand_start:
            start = m_cand_start.end()
            m_finish = re.search(r"(?im)^\s*Finish:\s*(true|false)\s*$", t)
            if m_finish:
                end = m_finish.start()
                candidate = t[start:end].strip("\n")
            else:
                candidate = t[start:].strip("\n")

        # 3) Finish
        m_finish = re.search(r"(?im)^\s*Finish:\s*(true|false)\s*$", t)
        finish = False
        if m_finish:
            finish = (m_finish.group(1).lower() == "true")

        return {"thought": thought, "candidate": candidate, "finish": finish}
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            if t.startswith("python"):
                t = t[len("python"):].lstrip()
        return t
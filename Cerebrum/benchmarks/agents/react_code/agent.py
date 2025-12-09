from typing import Dict, Any, List
import time
import re
import threading
import json

from cerebrum.llm.apis import llm_chat, llm_call_tool
from cerebrum.config.config_manager import config
from cerebrum.interface import AutoTool
from litellm import completion

aios_kernel_url = config.get_kernel_url()


class ReActAgent:
    """
    ReActAgent with CodeTestRunner tool

    - Input: HumanEval-style task_input (function skeleton + docstring, no tests)
    - Loop:
        Observation (from history + tool output)
        Thought      (how to change code/tests)
        Action       (produce function body + tests, call tool)
    - LLM 출력은 plain text:
        Observation: ...
        Thought: ...
        Action:
        <FINAL_ANSWER>
            ... function body ...
        </FINAL_ANSWER>
        <TESTS>
        ... python tests ...
        </TESTS>
        Finish: yes|no
    - Tool: code/code_test_runner
      * code  : task_input + override def + body
      * tests : LLM이 만든 테스트 코드
    - 종료: tool Status == success AND Finish == yes
    - 반환: 마지막 Candidate (즉 <FINAL_ANSWER> ... </FINAL_ANSWER> 블록)
    """

    def __init__(self, on_aios: bool = True, max_steps: int = 3):
        self.agent_name = "react"
        self.on_aios = on_aios
        self.max_steps = max_steps

        self.model = "gpt-4o-mini"#"meta-llama/Llama-3.1-8B-Instruct"#"gpt-4o-mini"#   # 예: gpt-4o-mini / qwen3 등
        self.backend = "openai"#"vllm"#"openai"                             # openai / ollama / vllm
        self.llms = [{"name": self.model, "backend": self.backend}]

        self.history: List[Dict[str, Any]] = []

        # --- CodeTestRunner tool 준비 ---
        self.tool = AutoTool.from_preloaded("code/code_test_runner")
        self.tools = [self.tool.get_tool_call_format()]

    # ----------------------------------------------------------------------
    #                           MAIN RUN LOOP
    # ----------------------------------------------------------------------
    def run(self, task_input: str) -> str:
        """
        task_input: HumanEval-style prompt (function skeleton + docstring)
        returns: last Candidate string (with <FINAL_ANSWER> wrapper)
        """
        self.history.clear()

        # task_input에서 함수 헤더(def ...) 추출
        func_header = self._extract_def_header(task_input)
        if not func_header:
            raise ValueError("Could not find function header (def ...) in task_input.")

        # System prompt: Observation / Thought / Action + tool 사용 규칙
        system_prompt = f"""
You are a senior Python assistant solving a HumanEval-style code-completion task
using a compact ReAct loop (Observation → Thought → Action) and the tool "code_test_runner".

CONTEXT:
- Input is a Python file fragment that defines ONE target function with a docstring.
- There are NO tests provided; you must create Python tests yourself.
- You CANNOT run code; an external tool executes the code + tests and reports results.
- You will see a short History summarizing previous rounds (status, exit code, errors).

YOUR LOOP:
1) Observation:
   - Summarize what happened in the latest tool run from History
     (e.g., success/failure, key assertion error, traceback hint).
2) Thought:
   - Explain briefly how you will change the implementation and/or tests.
   - Consider edge cases and correctness.
3) Action:
   - Propose a new function BODY (only the inside, NOT the 'def' line),
     wrapped EXACTLY in <FINAL_ANSWER> ... </FINAL_ANSWER>.
   - Propose Python tests that exercise the function, wrapped in <TESTS> ... </TESTS>.
   - Tests MUST be executable Python code using assert statements
     (do NOT use doctest syntax with >>>).

OUTPUT FORMAT (STRICT, PLAIN TEXT, NO MARKDOWN):

Observation: <one or two sentences about the latest tool result>
Thought: <one or two sentences about how you will change code/tests>
Action:
<FINAL_ANSWER>
    [your Python function body here, each line indented with EXACTLY 4 spaces]
</FINAL_ANSWER>
<TESTS>
[valid Python test code using 'assert', possibly multiple lines]
</TESTS>
Finish: <yes|no>

RULES:
- Do NOT include the 'def' line or docstring in <FINAL_ANSWER>; only the function body.
- Each code line inside <FINAL_ANSWER> MUST begin with exactly 4 spaces.
- In <TESTS>, call the target function using the name and parameters from the task input.
- Always include at least the examples from the docstring as assert-based tests.
- After each Action, the environment will:
  * Build full code from the original task_input + your function body override,
  * Run your tests via the CodeTestRunner tool,
  * Append a summary of Status / Exit code / error head to History.
- Use that History to update your Observation in the next round.
- If you believe your current Candidate passes all relevant tests,
  set Finish: yes. Otherwise, set Finish: no.

Task:
{task_input}
""".strip()

        messages = [{"role": "system", "content": system_prompt}]

        final_candidate = ""

        for step in range(self.max_steps):
            # --- Step prompt: 짧은 history 요약만 제공 (tool 결과 반영) ---
            step_prompt = f"""History (latest first, up to 3):
{self._format_history_for_prompt(self.history[-3:])}

Now continue the ReAct loop and output EXACTLY:

Observation: ...
Thought: ...
Action:
<FINAL_ANSWER>
    ...
</FINAL_ANSWER>
<TESTS>
...
</TESTS>
Finish: <yes|no>"""

            messages.append({"role": "user", "content": step_prompt})

            # --- LLM 호출 ---
            raw = self._call_llm(messages)

            # --- plain-text 파싱 (Observation / Thought / Action / Finish) ---
            step_obj = self._parse_react_step(raw)
            observation = step_obj["observation"]
            thought = step_obj["thought"]
            candidate = step_obj["candidate"]     # <FINAL_ANSWER> ... 포함
            body = step_obj["body"]               # 실제 함수 몸체 (4-space indent)
            tests = step_obj["tests"]             # <TESTS> ... 안의 내용 (Python 코드)
            finish_flag = step_obj["finish"]      # bool (yes → True)

            final_candidate = candidate  # 항상 최신 Candidate를 최종 후보로 유지

            #breakpoint()

            # --- Code + Tests 구성해서 tool 실행 ---
            full_code = self._build_full_code(task_input, func_header, body)
            worker_params = {
                "code": full_code,
                "tests": tests,
                "timeout": 5.0,
            }

            tool_status, exit_code, err_head, raw_tool_msg = self._run_tool(worker_params)

            # --- history에 tool 결과 반영 ---
            self.history.append(
                {
                    "round": step,
                    "observation": observation,
                    "thought": thought,
                    "finish": finish_flag,
                    "status": tool_status,
                    "exit_code": exit_code,
                    "error_head": err_head,
                }
            )

            #breakpoint()

            # --- 종료 조건 ---
            # Tool이 success AND LLM이 Finish: yes 라고 한 경우에만 종료
            #if tool_status == "success" and finish_flag and final_candidate:
            if tool_status == "success":#and finish_flag == True:
                break

        return final_candidate

    # ----------------------------------------------------------------------
    #                           LLM 호출 헬퍼
    # ----------------------------------------------------------------------
    def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        if self.on_aios:
            resp = llm_chat(
                agent_name=self.agent_name,
                messages=messages,
                base_url=aios_kernel_url,
                llms=self.llms,
            )
            raw = resp["response"]["response_message"]
            return raw or ""

        # non-AIOS (직접 vLLM / OpenAI 서버 호출)
        if self.model == "meta-llama/Llama-3.1-8B-Instruct":
            non_aios_resp = completion(
                model="hosted_vllm/" + self.model,
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
            return non_aios_resp or ""

        raw = (
            non_aios_resp.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )
        if raw is None:
            raw = non_aios_resp.get("content", "")
        return raw or ""

    # ----------------------------------------------------------------------
    #                          TOOL 호출 & 요약
    # ----------------------------------------------------------------------
    def _run_tool(self, worker_params: Dict[str, Any]):
        """
        CodeTestRunner tool 호출 + 결과 요약

        - on_aios = True  → AIOS ToolHub (llm_call_tool) 사용
        - on_aios = False → preloaded AutoTool(code/code_test_runner)를 직접 호출
        """

        # -------------------------
        # 1) non-AIOS: AutoTool 인스턴스를 직접 실행
        # -------------------------
        if not self.on_aios:
            try:
                # AutoTool은 내부적으로 BaseTool(CodeTestRunner)을 감싸고 있고,
                # run(params) 형태로 호출된다고 가정
                raw_msg = self.tool.run(worker_params)
            except Exception as e:
                # 예외도 항상 같은 포맷으로 만들어서 위에서 그대로 파싱 가능하게
                raw_msg = (
                    "Status: failure\n"
                    "Exit code: -1\n\n"
                    "[STDOUT]\n\n"
                    "[STDERR]\n"
                    f"Exception while running CodeTestRunner locally: {e!r}"
                )

            status, exit_code, err_head = self._summarize_tool_output(raw_msg)
            return status, exit_code, err_head, raw_msg

        # -------------------------
        # 2) AIOS 모드: 기존 llm_call_tool 경로
        # -------------------------
        tool_messages = [
            {
                "role": "system",
                "content": (
                    "You are a tool-calling assistant. "
                    "Use the provided tool when appropriate."
                ),
            },
            {
                "role": "user",
                "content": "Run the tests for the generated solution.",
            }
        ]

        try:
            tool_resp = llm_call_tool(
                agent_name=self.agent_name,
                messages=tool_messages,
                tools=self.tools,
                base_url=aios_kernel_url,
                llms=self.llms,
            )["response"]
            raw_msg = tool_resp.get("response_message", "") or ""
        except Exception as e:
            raw_msg = (
                "Status: failure\n"
                "Exit code: -1\n\n"
                "[STDOUT]\n\n"
                "[STDERR]\n"
                f"Exception while calling CodeTestRunner via AIOS: {e!r}"
            )

        status, exit_code, err_head = self._summarize_tool_output(raw_msg)
        return status, exit_code, err_head, raw_msg

    @staticmethod
    def _summarize_tool_output(msg: str):
        """
        CodeTestRunner의 response_message에서
        - status: success/failure/timeout/unknown
        - exit_code
        - stderr 앞부분
        추출
        """
        status_match = re.search(r"Status:\s*(\w+)", msg)
        status = status_match.group(1).lower() if status_match else "unknown"

        exit_match = re.search(r"Exit code:\s*(\d+)", msg)
        exit_code = int(exit_match.group(1)) if exit_match else None

        stderr_head = ""
        stderr_match = re.search(r"\[STDERR\]\n(.*)", msg, re.DOTALL)
        if stderr_match:
            stderr_full = stderr_match.group(1).strip()
            lines = stderr_full.splitlines()
            stderr_head = "\n".join(lines[:5])

        return status, exit_code, stderr_head

    # ----------------------------------------------------------------------
    #                        HISTORY 포맷 (프롬프트용)
    # ----------------------------------------------------------------------
    @staticmethod
    def _format_history_for_prompt(hist: List[Dict[str, Any]]) -> str:
        if not hist:
            return "[]"
        lines = []
        for h in reversed(hist):
            err = (h.get("error_head") or "").replace("\n", " ")[:80]
            lines.append(
                f"- Round {h['round']}: status={h.get('status')}, "
                f"exit={h.get('exit_code')}, finish={h.get('finish')}, "
                f"obs='{h.get('observation','')[:40]}', err_head='{err}'"
            )
        return "\n".join(lines)

    # ----------------------------------------------------------------------
    #                       ReAct step plain-text 파서
    # ----------------------------------------------------------------------
    @staticmethod
    def _parse_react_step(text: str) -> Dict[str, Any]:
        """
        Expect LLM output:

        Observation: ...
        Thought: ...
        Action:
        <FINAL_ANSWER>
            ...
        </FINAL_ANSWER>
        <TESTS>
        ...
        </TESTS>
        Finish: yes|no
        """
        t = text.replace("\r\n", "\n").replace("\r", "\n")

        # Observation
        m_obs = re.search(r"(?im)^\s*Observation:\s*(.*)$", t)
        observation = m_obs.group(1).strip() if m_obs else ""

        # Thought
        m_th = re.search(r"(?im)^\s*Thought:\s*(.*)$", t)
        thought = m_th.group(1).strip() if m_th else ""

        # Candidate body (within <FINAL_ANSWER>...</FINAL_ANSWER>)
        m_body = re.search(
            r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", t, re.DOTALL | re.IGNORECASE
        )
        if not m_body:
            body = ""
            candidate = ""
        else:
            body = m_body.group(1).strip("\n")
            candidate = f"<FINAL_ANSWER>\n{body}\n</FINAL_ANSWER>"

        # Tests (within <TESTS>...</TESTS>)
        m_tests = re.search(
            r"<TESTS>(.*?)</TESTS>", t, re.DOTALL | re.IGNORECASE
        )
        tests = m_tests.group(1).strip("\n") if m_tests else ""

        # Finish
        m_fin = re.search(r"(?im)^\s*Finish:\s*(yes|no)\s*$", t)
        finish = False
        if m_fin:
            finish = m_fin.group(1).lower() == "yes"

        return {
            "observation": observation,
            "thought": thought,
            "candidate": candidate,
            "body": body,
            "tests": tests,
            "finish": finish,
        }

    # ----------------------------------------------------------------------
    #                         코드 빌드 헬퍼
    # ----------------------------------------------------------------------
    @staticmethod
    def _extract_def_header(task_input: str) -> str:
        """
        task_input에서 첫 번째 'def ' 라인을 함수 헤더로 추출.
        (공백 제거한 라인으로 반환)
        """
        for line in task_input.splitlines():
            if line.strip().startswith("def "):
                return line.strip()
        return ""

    @staticmethod
    def _build_full_code(task_input: str, func_header: str, body: str) -> str:
        """
        CodeTestRunner에 넘길 full code 생성.

        전략:
        - 원래 task_input 전체를 그대로 script 상단에 둔다. (기존 skeleton 유지)
        - 그 아래에 "override" 함수 정의를 다시 한 번 작성한다.
          (동일한 이름의 함수는 나중에 정의된 것이 바인딩되므로, override 동작)
        - body는 <FINAL_ANSWER> 안의 내용으로, 이미 4-space indent라고 가정.

        script:
            [task_input 그대로]
            # --- Solution override ---
            def ...same header...:
                ... body ...
        """
        script_lines = []
        script_lines.append(task_input.rstrip("\n"))
        script_lines.append("\n\n# --- Solution override ---")
        script_lines.append(func_header.rstrip(":") + ":")
        script_lines.append(body)  # body는 이미 줄마다 4 space indentation

        return "\n".join(script_lines) + "\n"


# ----------------------------------------------------------------------
#                              간단 테스트
# ----------------------------------------------------------------------
def main():
    agent = ReActAgent()

    task_input = """
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
    pass
""".strip()

    result = agent.run(task_input)
    print("\n=== FINAL CANDIDATE ===")
    print(result)


if __name__ == "__main__":
    main()

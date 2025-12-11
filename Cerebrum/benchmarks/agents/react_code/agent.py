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

HistoryEntry = Dict[str, Any]


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
    - 종료: tool Status == success AND Finish == yes (현재는 success 만으로 종료)
    - 반환: 마지막 Candidate (즉 <FINAL_ANSWER> ... </FINAL_ANSWER> 블록)
    """

    # ------------------------------------------------------------------
    # 정규식 패턴 (한 번만 컴파일해서 재사용)
    # ------------------------------------------------------------------
    _OBS_PATTERN = re.compile(r"(?im)^\s*Observation:\s*(.*)$")
    _THOUGHT_PATTERN = re.compile(r"(?im)^\s*Thought:\s*(.*)$")
    _FINISH_PATTERN = re.compile(r"(?im)^\s*Finish:\s*(yes|no)\s*$")
    _FINAL_PATTERN = re.compile(
        r"<FINAL_ANSWER>(.*?)</FINAL_ANSWER>", re.DOTALL | re.IGNORECASE
    )
    _TESTS_PATTERN = re.compile(
        r"<TESTS>(.*?)</TESTS>", re.DOTALL | re.IGNORECASE
    )

    def __init__(self, on_aios: bool = True, max_steps: int = 3):
        self.agent_name = "react"
        self.on_aios = on_aios
        self.max_steps = max_steps

        self.model = "meta-llama/Llama-3.1-8B-Instruct"  # gpt-4o-mini / meta-llama/Llama-3.1-8B-Instruct
        self.backend = "vllm"  # openai / vllm
        self.llms = [{"name": self.model, "backend": self.backend}]

        self.history: List[HistoryEntry] = []

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
        system_prompt = self._build_system_prompt(task_input)
        messages = [{"role": "system", "content": system_prompt}]

        final_candidate = ""

        for step in range(self.max_steps):
            # --- Step prompt: 짧은 history 요약만 제공 (tool 결과 반영) ---
            step_prompt = f"""History (latest first, up to 3):
{self._format_history_for_prompt(self.history[-3:])}

{self._build_step_prompt()}"""

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

            # breakpoint()

            # --- Code + Tests 구성해서 tool 실행 ---
            full_code = self._build_full_code(task_input, func_header, body)
            worker_params = {
                "code": full_code,
                "tests": tests,
                "timeout": 5.0,
            }

            tool_status, exit_code, err_head, raw_tool_msg = self._run_tool(worker_params)

            # breakpoint()

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

            # --- 종료 조건 ---
            # 현재 정책: 툴 결과가 success면 Finish 플래그와 상관없이 종료.
            # (Finish까지 보려면 아래 주석 처리된 조건으로 바꾸면 됨.)
            # if tool_status == "success" and finish_flag and final_candidate:
            if tool_status == "success":
                break

        return final_candidate

    # ----------------------------------------------------------------------
    #                      프롬프트 빌더 (System / Step)
    # ----------------------------------------------------------------------
    def _build_system_prompt(self, task_input: str) -> str:
        return f"""
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

    @staticmethod
    def _build_step_prompt() -> str:
        return """Now continue the ReAct loop and output EXACTLY:

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
        if self.on_aios:
            raw_msg = self._run_tool_aios(worker_params)
        else:
            raw_msg = self._run_tool_local(worker_params)

        status, exit_code, err_head = self._summarize_tool_output(raw_msg)
        return status, exit_code, err_head, raw_msg

    def _run_tool_local(self, worker_params: Dict[str, Any]) -> str:
        """
        non-AIOS 모드: AutoTool 인스턴스를 직접 실행
        """
        try:
            return self.tool.run(worker_params)
        except Exception as e:
            return (
                "Status: failure\n"
                "Exit code: -1\n\n"
                "[STDOUT]\n\n"
                "[STDERR]\n"
                f"Exception while running CodeTestRunner locally: {e!r}"
            )

    def _run_tool_aios(self, worker_params: Dict[str, Any]) -> str:

        code = worker_params.get("code", "")
        tests = worker_params.get("tests", "")
        timeout = float(worker_params.get("timeout", 5.0))

        tool_messages = [
            {
                "role": "system",
                "content": (
                    "You are a tool-calling assistant.\n"
                    "You have access to a single tool named `code/code_test_runner`.\n"
                    "\n"
                    "This tool executes Python code together with test code in an isolated subprocess. "
                    "It expects a JSON object with the following fields as its arguments:\n"
                    "  - `code`   (string): FULL Python source code to be tested.\n"
                    "  - `tests`  (string): FULL Python test code that uses the definitions in `code`.\n"
                    "  - `timeout` (number, optional): timeout in seconds for running the tests.\n"
                    "\n"
                    "CRITICAL RULES:\n"
                    "1. Do NOT pass file names or file paths (e.g. 'solution.py', 'test_solution.py').\n"
                    "2. Instead, use the RAW code strings that are provided between <CODE> and <TESTS>.\n"
                    "3. Do NOT summarize, shorten, or modify the code or tests; copy them EXACTLY.\n"
                    "4. The `timeout` argument MUST be a NUMBER (e.g. 5 or 5.0), NOT a string.\n"
                    "5. When calling the tool, construct arguments like:\n"
                    "   {\"code\": <string>, \"tests\": <string>, \"timeout\": <number>}.\n"
                    "\n"
                    "Your only job is to call `code/code_test_runner` ONCE with the correct arguments."
                )
            },
            {
                "role": "user",
                "content": (
                    "Run the tool `code/code_test_runner` on the following code and tests.\n\n"
                    f"Use timeout = {timeout}.\n\n"
                    "<CODE>\n"
                    f"{code}\n"
                    "</CODE>\n\n"
                    "<TESTS>\n"
                    f"{tests}\n"
                    "</TESTS>\n"
                    "\n"
                    "Call the tool now using these exact strings for `code` and `tests`."
                )
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
            #breakpoint()
            return tool_resp.get("response_message", "") or ""
            
        except Exception as e:
            return (
                "Status: failure\n"
                "Exit code: -1\n\n"
                "[STDOUT]\n\n"
                "[STDERR]\n"
                f"Exception while calling CodeTestRunner via AIOS: {e!r}"
            )
        

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
    def _format_history_for_prompt(hist: List[HistoryEntry]) -> str:
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
        m_obs = ReActAgent._OBS_PATTERN.search(t)
        observation = m_obs.group(1).strip() if m_obs else ""

        # Thought
        m_th = ReActAgent._THOUGHT_PATTERN.search(t)
        thought = m_th.group(1).strip() if m_th else ""

        # Candidate body (within <FINAL_ANSWER>...</FINAL_ANSWER>)
        m_body = ReActAgent._FINAL_PATTERN.search(t)
        if not m_body:
            body = ""
            candidate = ""
        else:
            body = m_body.group(1).strip("\n")
            candidate = f"<FINAL_ANSWER>\n{body}\n</FINAL_ANSWER>"

        # Tests (within <TESTS>...</TESTS>)
        m_tests = ReActAgent._TESTS_PATTERN.search(t)
        tests = m_tests.group(1).strip("\n") if m_tests else ""

        # Finish
        m_fin = ReActAgent._FINISH_PATTERN.search(t)
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
        #script_lines.append("\n\n# --- Solution override ---")
        #script_lines.append(func_header.rstrip(":") + ":")
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

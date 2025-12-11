from typing import Dict, Any, List
import time
import re
import threading
import json

from cerebrum.llm.apis import llm_chat_with_json_output
from cerebrum.utils import _parse_json_output

from cerebrum.config.config_manager import config
from cerebrum.interface import AutoTool

from litellm import completion



aios_kernel_url = config.get_kernel_url()

HistoryEntry = Dict[str, Any]


class ReActAgent:
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
        self.history_window = 1

        self.model = "meta-llama/Llama-3.1-8B-Instruct"  # gpt-4o-mini / meta-llama/Llama-3.1-8B-Instruct
        self.backend = "vllm"  # openai / vllm
        self.llms = [{"name": self.model, "backend": self.backend}]

        self.history = []

        self.tool = AutoTool.from_preloaded("code/code_test_runner")
        #self.tools = [self.tool.get_tool_call_format()]

    # ----------------------------------------------------------------------
    #                           MAIN RUN LOOP
    # ----------------------------------------------------------------------
    def run(self, task_input: str) -> str:
        self.history.clear()

        func_header = self._extract_def_header(task_input)
        if not func_header:
            raise ValueError("Could not find function header (def ...) in task_input.")

        system_prompt = f"""
You are a Python coding agent that iteratively improves a solution so that it passes the given tests.
When producing Python code, always format it using proper indentation and newlines.
Each statement must appear on its own line. Never compress multiple statements into one line.
You must complete a given Python function(function header and docstring) : 
{task_input}

You must reply with one valid JSON object and nothing else.
The JSON must have exactly the following fields:

- "observation": your short summary of the current situation or test results.
- "reasoning": your reasoning about what to do next.
- "action": one of "run", or "finish".
- "tool_params": parameters for the test tool

Semantics:
- "run":      Request the environment to run the test tool with the given "tool_params".
- "finish":   You believe the current solution is final and correct; stop the loop.

When "action" == "run", you must set "tool_params" to an object of the form:
{{
  "code":  "Python solution code to test",
  "tests": "Python test code to run with that solution"
}}
When "action" is "finish", you must set "tool_params" to null.

Examples of valid outputs:
{{
  "observation": "All tests have passed in the most recent run.",
  "reasoning": "The current implementation satisfies all test cases, so I will stop iterating and accept this solution as final.",
  "action": "finish",
  "tool_params": null
}}

{{
  "observation": "Previous tests failed with an IndexError.",
  "reasoning": "I fixed the loop bounds and now want to rerun the tests.",
  "action": "run",
  "tool_params": {{
    "code": "    return number - int(number)",
    "tests": "assert truncate_number(3.5) == 0.5"
  }}
}}
"""
        messages = [{"role": "system", "content": system_prompt}]

        final_candidate = ""

        for step in range(self.max_steps):
            step_prompt = f"""Step-by-Step Execution Protocol

Here are the latest {self.history_window} steps (at most) you have taken:
<history>
{self.history[-self.history_window:]}
</history>

Use the system instructions and the above history to decide what to do in this step.

Your tasks in this step:

1. Observation:
   - Look at the **most recent entry in the history**, especially its `status` and `stderr`
   - If there is no history yet, your observation should reflect that (e.g., "No tests have been run yet.").
   Examples:
     - "status=success, no errors in stderr."
     - "status=failure, stderr starts with: AssertionError: expected True but got False."
     - "No tests have been run yet."

2. Reasoning:
   - Based on your observation, think about **what is wrong in the current solution** and **how to fix it**.
   - Use the error messages (e.g., AssertionError, IndexError, wrong return value) to infer which part of the logic, boundary condition, or special case is broken.
   - In the "reasoning" field, clearly explain:
       - what you think the bug or issue is, and
       - what kind of change to the code and/or tests is needed to move closer to passing all tests.

3. Action:
   - Based on your observation and reasoning, decide what to do next and set the "action" field to one of:
       - "run": you are ready to test a concrete version of the solution; ask the environment to run the test tool.
       - "finish": you believe the current solution is sufficient and correct; stop the loop.

   - When "action" == "run":
       - You must set "tool_params" to an object of the form:
         {{
           "code":  "function body of solution code to test",
           "tests": "Python test code to run with that solution"
         }}
       - This is where you actually materialize the solution code and the tests you want to run.

    - IMPORTANT: The external tool will build **one single Python file** in the following order:
        1) the existing function header and docstring (already given in the prompt),
        2) your "code" (function body),
        3) your "tests".

        Conceptually, the file will look like:

        def truncate_number(number: float) -> float:
            \"\"\" existing docstring ... \"\"\"
            <your code goes here, as the indented function body>

        <your tests go here in the same file>

    - Therefore, **"code" must be ONLY the function body**, not a full function definition:
        - Do NOT repeat the `def ...` line.
        - Do NOT add imports or top-level code here.
        - You MUST include the correct indentation so that the body fits directly under the existing function definition.
        Example of a valid "code" value:
        "    return number - int(number)"

    - The "tests" field must be Python code that will be appended after the solution in the same file.
    It should consist of one or more assert statements that directly call the completed function.
    For example:
        assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True
        assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False

Output requirements (must match the system prompt):
Return ONLY the JSON object.
Do NOT wrap it in markdown fences.
Do NOT add any extra text before or after the JSON."""

            messages.append({"role": "user", "content": step_prompt})

            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "orchestration",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "observation": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "action": {
                                "type": "string",
                                "enum": ["continue", "use", "finish"],
                            },
                            "tool_params": {
                                "anyOf": [
                                    {"type": "null"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "code": {"type": "string"},
                                            "tests": {"type": "string"},
                                        },
                                        "required": ["code", "tests"],
                                    },
                                ]
                            },
                        },
                        "required": ["observation", "reasoning", "action", "tool_params"],
                    },
                },
            }

            response = llm_chat_with_json_output(
                agent_name=self.agent_name,
                messages=messages,
                llms=self.llms,
                response_format=response_format
            )

            #breakpoint()

            step_response = response["response"]["response_message"]
            
            resp_dict = _parse_json_output(step_response)
            observation = resp_dict.get("observation", "")
            reasoning = resp_dict.get("reasoning", "")
            action = resp_dict.get("action", None)
            tool_params = resp_dict.get("tool_params") or {}

            final_candidate = tool_params.get("code", "")

            params = {
                "header": task_input,
                "code": tool_params.get("code", ""),
                "tests": tool_params.get("tests", ""),
            }

            #breakpoint()

            if action == "run":
                result = self.tool.run(params)
                if isinstance(result, dict):
                    status = result.get("status", "")
                    stderr = result.get("stderr", "")
            else:
                status = "error"
                stderr = str(result)

            # breakpoint()

            self.history.append(
                {
                    "round": step,
                    "observation": observation,
                    "reasoning": reasoning,
                    "action": action,
                    "status": status,
                    "stderr": stderr,
                }
            )

            #if status == "success":
            if action == "finish":
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

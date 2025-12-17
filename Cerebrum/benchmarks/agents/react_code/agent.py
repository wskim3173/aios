from typing import Dict, Any, List

import json
import re
import textwrap

from cerebrum.llm.apis import llm_chat_with_json_output
from cerebrum.utils import _parse_json_output
from cerebrum.config.config_manager import config
from cerebrum.interface import AutoTool

from typing import Tuple
from litellm import completion

aios_kernel_url = config.get_kernel_url()

class ReActAgent:
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
    # ----------------------------------------------------------------------
    #                           MAIN RUN LOOP
    # ----------------------------------------------------------------------
    def run(self, task_input: str) -> str:
        self.history.clear()

        system_prompt = f"""
    You are a Python coding agent following a ReAct loop:
    Observation -> Reasoning -> Action (run tool OR finish).

    ## Task
    You must solve the programming task described in <function_definition>.
    You will iteratively propose a FULL Python function implementation, run tests via the tool,
    then use the tool result (status/stderr) to improve the next attempt.

    <function_definition>
    {task_input}
    </function_definition>

    ## Output (STRICT)
    Return ONLY one valid JSON object with EXACTLY these fields:
    - "observation": short factual summary of the latest situation
    - "reasoning": what to change next and why
    - "action": one of ["run", "finish"]
    - "answer": a SINGLE string containing BOTH tagged sections:

    <CODE>
    ... full Python code including the TARGET function definition (def ...:) ...
    </CODE>
    <TESTS>
    ... assert-based tests that call the target function ...
    </TESTS>

    ## Formatting rules (VERY IMPORTANT)
    1) The answer string MUST include EXACTLY ONE <CODE>...</CODE> block
    and EXACTLY ONE <TESTS>...</TESTS> block. Use proper closing tags.
    Do NOT add any extra tags. Do NOT output "CODE ..." or "TESTS ..." as plain text. 
    Use literal \\n for newlines inside the string.
    
    2) Inside <CODE> (STRICT — full function this time):
    - You MUST include the complete function definition line: def ...:
    - The function name and parameters MUST match the one in <function_definition>.
    - Avoid unnecessary imports. (The environment will prepend header/imports already.)
    - The function header MUST be on its own line, ending with ":" and NOTHING after the colon.
    - Valid:  def f(x):\n    ...
    - Invalid: def f(x): return x+1
    - Invalid: def f(x): if x<0: return 0 else: return 1
    - The function body MUST start on the next line and use a normal indented block (4 spaces).
    - Do NOT use one-line function bodies.
    - Do NOT put "if/for/while/return" on the same line as the def header.

    3) Inside <TESTS>:
    - Inside <TESTS>, you MUST write ONLY plain Python `assert` statements.
    - Provide at least 2 assert statements.
    - Tests MUST directly call the target function.
    - NO imports (no `import unittest`, no `from typing import ...`)
    - Keep tests small/fast (no large loops, no randomness, no I/O).

    4) Example:
    - Valid answer:
    "<CODE>\ndef add(a, b):\n    return a + b\n</CODE>\n<TESTS>\nassert add(1, 2) == 3\nassert add(0, 0) == 0\nassert add(-1, 1) == 0\n</TESTS>"

    - Invalid answer:
    "CODE def add(a,b): return a+b\nTESTS assert add(1,2)==3"
    """.strip()

        messages = [{"role": "system", "content": system_prompt}]
        final_code = ""

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "orchestration",
                "schema": {
                    "type": "object",
                    "properties": {
                        "observation": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "action": {"type": "string", "enum": ["run", "finish"]},
                        "answer": {"type": "string"},
                    },
                    "required": ["observation", "reasoning", "action", "answer"],
                    "additionalProperties": False,
                },
            },
        }

        for step in range(self.max_steps):
            step_prompt = f"""
    You must follow the ReAct loop and use the history below.

    <history>
    {self.history[-self.history_window:]}
    </history>

    ## Step instructions
    1) Observation:
    - If history is empty: say no tests have been run yet.
    - Otherwise: summarize last status and the FIRST LINE of stderr (if any).

    2) Reasoning:
    - If last status != success: explain what failed and how you will fix it.
    - You MUST modify the next <CODE> and/or <TESTS> based on that failure.
    - Prefer fixing CODE; adjust TESTS only to improve coverage for the spec/edge cases.

    3) Action:
    - Choose "finish" only when you feel confident the current solution is correct and you believe another tool run is unnecessary.
    - Otherwise choose "run" and improve <CODE>/<TESTS> based on the latest status, stderr.

    4) Mandatory fixes:
    - If your previous <CODE> used a one-line function like "def f(...): return ...",
    you MUST rewrite it into a multi-line function block:
    def f(...):\n    ...
    - If tests contained anything other than plain assert statements, you MUST replace them with assert-only tests.

    Now output ONE JSON object only.
    """.strip()

            messages.append({"role": "user", "content": step_prompt})

            raw = self._call_llm(messages=messages, response_format=response_format)

            # Parse JSON (minimal surface area)
            try:
                resp_dict = _parse_json_output(raw)
            except Exception:
                resp_dict = {
                    "observation": "Model did not return valid JSON; attempting tag-parse fallback.",
                    "reasoning": "I will extract <CODE> and <TESTS> from the raw output and run the tool.",
                    "action": "run",
                    "answer": raw if isinstance(raw, str) else "",
                }

            observation = resp_dict.get("observation", "")
            reasoning = resp_dict.get("reasoning", "")
            action = resp_dict.get("action", "run")
            answer = resp_dict.get("answer", "")

            code, tests = self._extract_code_tests_from_answer(answer)

            # Safety guard: don't allow finish before we ever got a success
            last_success = bool(self.history) and (self.history[-1].get("status") == "success")
            if action == "finish" and not last_success:
                action = "run"
            if action == "finish" and last_success:
                break

            # Run tool
            status = "failure"
            stderr = ""
            try:
                prelude = self._extract_prelude_from_task_input(task_input)
                ok, err = self.tool.run({"header": prelude, "code": code, "tests": tests})
                status = "success" if ok else "failure"
                stderr = err or ""
            except Exception as e:
                status = "exception"
                stderr = repr(e)

            # Update final_code policy
            if not final_code.strip():
                final_code = code
            elif status == "success":
                final_code = code

            # Record history for next step
            self.history.append(
                {
                    "round": step,
                    "observation": observation,
                    "reasoning": reasoning,
                    "action": action,
                    "code": code,
                    "tests": tests,
                    "status": status,
                    "stderr": stderr,
                }
            )

        #breakpoint()

        result = self._strip_def_and_normalize_body(final_code)

        if result == "":
            return final_code
        else:
            return result

    def _strip_def_and_normalize_body(self, llm_code: str) -> str:
        """
        If llm_code contains a full function (possibly with decorators and multi-line signature),
        remove the function header and return ONLY the function body, normalized to 4-space indentation.
        """
        ends_with_newline = llm_code.endswith("\n")
        s = llm_code.replace("\r\n", "\n").replace("\r", "\n")

        lines = s.split("\n")
        if not lines:
            return llm_code

        # 1) Find the start of the first function header (skip decorators / leading junk)
        def_start = None
        def_start_re = re.compile(r"^\s*(?:async\s+def|def)\s+\w+\s*\(")
        for i, line in enumerate(lines):
            if def_start_re.match(line):
                def_start = i
                break

        # If no def found, return as-is
        if def_start is None:
            return llm_code

        # 2) Find where the function header ends (supports multi-line signatures)
        # Track parentheses balance until it closes, and the line ends with ":" (optionally with comment).
        paren = 0
        header_end = None
        header_end_re = re.compile(r":\s*(#.*)?$")

        for j in range(def_start, len(lines)):
            line = lines[j]

            # crude but effective: count parens (good enough for typical LLM outputs)
            paren += line.count("(")
            paren -= line.count(")")

            if paren <= 0 and header_end_re.search(line.strip()):
                header_end = j
                break

        # If we failed to find header end, assume single-line header at def_start
        if header_end is None:
            header_end = def_start

        # 3) Everything after header_end is treated as body
        body = "\n".join(lines[header_end + 1:])

        # Normalize tabs -> spaces before dedent
        body = body.replace("\t", "    ")

        # Dedent and then re-indent with exactly 4 spaces for non-empty lines
        body = textwrap.dedent(body).strip("\n")

        fixed_lines = []
        for ln in body.splitlines():
            fixed_lines.append(("    " + ln) if ln.strip() else ln)

        out = "\n".join(fixed_lines)
        if ends_with_newline:
            out += "\n"
        return out

    def _extract_prelude_from_task_input(self, task_input: str) -> str:
        """
        Returns only the lines BEFORE the first top-level 'def ...('.
        This removes the function skeleton from header so the model can provide full def in <CODE>.
        """
        if not isinstance(task_input, str):
            return ""

        s = task_input.replace("\r\n", "\n").replace("\r", "\n")
        lines = s.split("\n")

        prelude = []
        for line in lines:
            # first function definition at top level
            if re.match(r"^\s*def\s+\w+\s*\(", line):
                break
            prelude.append(line)

        out = "\n".join(prelude).rstrip()
        return out + ("\n" if out else "")

    def _extract_code_tests_from_answer(self, answer: str) -> Tuple[str, str]:
        """
        Extract <CODE>...</CODE> and <TESTS>...</TESTS> from answer.
        Accept both actual newlines and literal '\\n' sequences.
        """
        if not isinstance(answer, str):
            return "", ""

        code_m = re.search(r"<CODE>\s*(.*?)\s*</CODE>", answer, re.DOTALL | re.IGNORECASE)
        tests_m = re.search(r"<TESTS>\s*(.*?)\s*</TESTS>", answer, re.DOTALL | re.IGNORECASE)

        code = code_m.group(1) if code_m else ""
        tests = tests_m.group(1) if tests_m else ""

        code = code.replace("\\n", "\n").strip("\n")
        tests = tests.replace("\\n", "\n").strip("\n")
        return code, tests

    # ----------------------------------------------------------------------
    #                           LLM CALL HELPER
    # ----------------------------------------------------------------------
    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        response_format: Dict[str, Any],
    ) -> str:

        if self.on_aios:
            response = llm_chat_with_json_output(
                agent_name=self.agent_name,
                messages=messages,
                llms=self.llms,
                response_format=response_format
            )
            return response["response"]["response_message"]

        if self.model == "meta-llama/Llama-3.1-8B-Instruct":
            non_aios_resp = completion(
                model="hosted_vllm/" + self.model,
                messages=messages,
                base_url="http://127.0.0.1:8091/v1",
                temperature=0.2,
                response_format=response_format
            )
        else:
            non_aios_resp = completion(
                model=self.model,
                messages=messages,
                temperature=0.2,
                response_format=response_format
            )

        if isinstance(non_aios_resp, str):
            return non_aios_resp

        raw = (
            non_aios_resp.get("choices", [{}])[0]
            .get("message", {})
            .get("content")
        )

        if raw is None:
            raw = non_aios_resp.get("content", "")
        return raw

# ----------------------------------------------------------------------
#                              SIMPLE TEST
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

from typing import Dict, Any, List

import json

from cerebrum.llm.apis import llm_chat_with_json_output
from cerebrum.utils import _parse_json_output
from cerebrum.config.config_manager import config
from cerebrum.interface import AutoTool

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
1. Task
You are a Python coding agent that iteratively improves a solution so that it passes the given tests.
When producing Python code, always format it using proper indentation and newlines.
You must complete a given Python function:
<function_definition>
{task_input}
</function_definition>

2. Output
You must reply with one valid JSON object
The JSON must have exactly the following fields:

- "observation": your short summary of the current situation.
- "reasoning": your reasoning about what to do next.
- "action": one of "run" or "finish".
- "tool_params": parameters for the tool

Output requirements (must match the system prompt):
Return ONLY the JSON object.
Do NOT wrap it in markdown fences.
Do NOT add any extra text before or after the JSON.

3. Semantics
- "run": If you think running the tool is necessary
- "finish": You believe the current solution is final and correct; stop the loop.

When "action" == "run", you must set "tool_params" to an object of the form:
{{
  "code":  "Python solution code to test",
  "tests": "Python test code to run with that solution"
}}
When "action" is "finish", you must set "tool_params" to null.

5. Examples of valid outputs
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
            status = ""
            stderr = ""
            step_prompt = f"""
##Step-by-Step Execution Protocol.
Here are the latest {self.history_window} steps (at most) you have taken:
<history>
{self.history[-self.history_window:]}
</history>

Use the system instructions and the above history to decide what to do in this step.

Your tasks in this step:

1. Observation:
   - Look at the most recent entry in the history, especially its `status` and `stderr`
   - If there is no history yet, your observation should reflect that (e.g., "No tests have been run yet.").
   Examples:
     - "status=success, no errors in stderr."
     - "status=failure, stderr starts with: AssertionError: expected True but got False."
     - "No tests have been run yet."

2. Reasoning:
   - Based on your observation, think about what is wrong in the current solution and how to fix it.
   - Use the error messages to infer which part of the logic, boundary condition, or special case is broken.
   - In the "reasoning" field, clearly explain:
       - what you think the bug or issue is, and
       - what kind of change to the code and/or tests is needed to move closer to passing all tests.

3. Action:
   - Based on your observation and reasoning, decide what to do next and set the "action" field to one of:
       - "run": you are ready to test a concrete version of the solution
       - "finish": you believe the current solution is sufficient and correct

   - When "action" == "run":
       - You must set "tool_params" to an object of the form:
         {{
           "code":  "function body of solution code to test",
           "tests": "Python test code to run with that solution"
         }}
       - This is where you actually materialize the solution code and the tests you want to run.

 4. IMPORTANT:
    - "code" must be ONLY the function body, not a full function definition:
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
"""

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

            response = self._call_llm(messages=messages, response_format=response_format)

            #breakpoint()
            
            resp_dict = _parse_json_output(response)
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

            # breakpoint()

            self.history.append(
                {
                    "round": step,
                    "observation": observation,
                    "reasoning": reasoning,
                    "action": action,
                    #"code": params["code"],
                    #"tests": params["tests"],
                    "status": status,
                    "stderr": stderr,
                }
            )

            if status == "success":
            #if action == "finish":
                break

        return final_candidate

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

        non_aios_resp = None

        if self.model == "meta-llama/Llama-3.1-8B-Instruct":
            schema_str = json.dumps(response_format, indent=2)

            messages.append({
                "role": "system",
                "content": f"""
                You MUST output JSON strictly following this schema:

                {schema_str}
                """
            })

            non_aios_resp = completion(
                model="hosted_vllm/" + self.model,
                messages=messages,
                base_url="http://127.0.0.1:8091/v1",
                temperature=0.2,
                #response_format=response_format
            )
        else:
            non_aios_resp = completion(
                model=self.model,
                messages=messages,
                temperature=0.2,
                response_format=response_format
            )

        #breakpoint()

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

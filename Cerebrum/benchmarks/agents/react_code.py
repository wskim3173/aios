from cerebrum.llm.apis import (
    llm_chat,
    llm_call_tool,
    llm_chat_with_json_output,
)
from cerebrum.interface import AutoTool
from cerebrum.utils import _parse_json_output
from cerebrum.config.config_manager import config

import json
from typing import Dict, Any

aios_kernel_url = config.get_kernel_url()


class ReActAgent:
    """
    ReAct + Self-Refine + Tool Systemcall Version
    - Iteratively improves code until all tests pass.
    """

    def __init__(self, on_aios: bool = True):
        self.agent_name = "react"
        self.on_aios = on_aios

        self.model = "gpt-4o-mini"   #gpt-4o-mini #qwen3:1.7b #meta-llama/Llama-3.1-8B-Instruct
        self.backend = "openai"     #openai #ollama #vllm
        self.llms = [{"name": self.model, "backend": self.backend}]

        self.max_steps = 7
        self.history_window = 3
        self.history = []

        # Load tool schemas from ToolHub
        tool = AutoTool.from_preloaded("code/code_test_runner")
        schema = tool.get_tool_call_format()
        #schema["function"]["name"] = "code/code_test_runner"  # <-- VERY IMPORTANT
        self.tools = [schema]

    # -----------------------------------------------------------------------------------
    #                               MAIN RUN LOOP
    # -----------------------------------------------------------------------------------

    def run(self, task_input: str) -> str:
        
        system_prompt = """
    You are a senior Python assistant that fixes a single function using a ReAct loop
    and the tool \"code_test_runner\".

    - Input: a task that contains a Python function (possibly buggy) and its tests.
    - You CANNOT run code yourself; only the tool runs tests.
    - The environment gives you a short <history> with previous code, tool outputs,
    and your own reasoning.

    At each step you MUST do:

    1) observation
    - Read the latest test result in <history>.
    - Briefly state what happened (which tests failed, error messages, etc.).

    2) reasoning
    - Explain what is wrong in the current implementation.
    - Decide how to fix it (what logic to change, which edge cases to handle).

    3) action
    - Write a full revised implementation of the function (def + body).
    - If you want to run tests on this version:
        * set worker_name = "code_test_runner"
        * set worker_params = {
            "code":  "<NEW full implementation>",
            "tests": "<original tests>",
            "timeout": 5.0
            }
    - If you are sure that ALL tests have already passed in the latest tool run:
        * set worker_name = null
        * set worker_params = null

    Rules:
    - Do NOT invent test results; rely only on tool outputs in <history>.
    - Do NOT resend exactly the same code after a failing run; always refine it.
    - Put only the function implementation into worker_params["code"]; never put tests there.
    """

        messages = [{"role": "system", "content": system_prompt}]

        final_answer = ""
        rounds = 0

        # ------------------------------------------------------------------------------
        #                               LOOP
        # ------------------------------------------------------------------------------

        while rounds < self.max_steps:
            # Provide recent trajectory
            step_instructions = f"""
        Return the next ReAct step as a single JSON object:

        {{
        "observation": "...",
        "reasoning": "...",
        "worker_name": "code_test_runner" or null,
        "worker_params": {{
            "code": "<full updated implementation>",
            "tests": "<original tests>",
            "timeout": 5.0
        }} or null
        }}

        - "observation": what you learned from the latest test result in history.
        - "reasoning": how you will change the implementation and why.
        - To run tests on a new implementation:
            * worker_name = "code_test_runner"
            * worker_params = the updated full code and the original tests.
        - If, following the system instructions, you judge that all tests have already
        passed and no more changes are needed:
            * worker_name = null
            * worker_params = null.

        Use the following history as context:

        <history>
        {self.history[-self.history_window:]}
        </history>
        """

            messages.append({"role": "user", "content": step_instructions})

            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "react_step",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "observation": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "worker_name": {
                                "anyOf": [{"type": "string"}, {"type": "null"}]
                            },
                            "worker_params": {
                                "anyOf": [{"type": "object"}, {"type": "null"}]
                            },
                        },
                        "required": [
                            "observation",
                            "reasoning",
                            "worker_name",
                            "worker_params",
                        ],
                    },
                },
            }

            step_raw = llm_chat_with_json_output(
                agent_name=self.agent_name,
                messages=messages,
                llms=self.llms,
                response_format=response_format,
            )

            step = _parse_json_output(step_raw["response"]["response_message"])

            observation = step["observation"]
            reasoning = step["reasoning"]
            worker_name = step["worker_name"]
            worker_params = step["worker_params"]

            #breakpoint()

            # ----------------------------------------------------------------------
            #                     TERMINATION (tests passed)
            # ----------------------------------------------------------------------
            if worker_name is None:
                self.history.append(
                    {
                        "round": rounds,
                        "observation": observation,
                        "thought": reasoning,
                        "called_worker": None,
                        "called_worker_params": None,
                        "info": None,
                    }
                )
                final_answer = self.get_final_answer(task_input)
                break

            # ----------------------------------------------------------------------
            #                    EXECUTE TOOL SYSTEMCALL
            # ----------------------------------------------------------------------

            tool_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a tool-calling assistant. "
                        "You MUST call exactly one of the provided tools "
                        "using the JSON parameters below."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Call the tool with:\n{json.dumps(worker_params)}",
                },
            ]

            tool_resp = llm_call_tool(
                agent_name=self.agent_name,
                messages=tool_messages,
                tools=self.tools,
                base_url=aios_kernel_url,
            )["response"]

            # Append to history
            traj = {
                "round": rounds,
                "observation": observation,
                "thought": reasoning,
                "called_worker": worker_name,
                "called_worker_params": worker_params,
                "info": tool_resp,
            }
            #print(traj)
            self.history.append(traj)

            rounds += 1

        return final_answer

    # -----------------------------------------------------------------------------------
    #                                FINAL ANSWER
    # -----------------------------------------------------------------------------------

    def get_final_answer(self, task_input: str) -> str:
        system_prompt = """
    You are an extractor agent for a code-completion task.

    You are given:
    - <history>: the full ReAct trajectory, including the last code version that passed all tests.
    - <task>: the original problem description with the function signature and tests.

    Your job:
    - Find the final correct implementation of the function in the history.
    - Return only the FUNCTION BODY (without the 'def ...' line), preserving valid Python indentation.
    - Wrap the body exactly in the following tags and nothing else:

    <FINAL_ANSWER>
    ...function body...
    </FINAL_ANSWER>

    Rules:
    - Do not add explanations, comments, or any extra text.
    - Do not repeat the function signature.
    - Do not change the parameter names or behavior of the final implementation.
    - Do not include any test code.
    """

        prompt = f"""
    <history>{self.history}</history>

    <task>{task_input}</task>

    Extract the final correct implementation's FUNCTION BODY only, and return it exactly as:

    <FINAL_ANSWER>
    ...function body...
    </FINAL_ANSWER>
    """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        resp = llm_chat(
            agent_name=self.agent_name,
            messages=messages,
            llms=self.llms
        )

        return resp["response"]["response_message"]


# -----------------------------------------------------------------------------------------
#                                   MAIN TEST
# -----------------------------------------------------------------------------------------

def main():
    agent = ReActAgent()

    sample_code = """
def add(a, b):
    return a * b   # WRONG ON PURPOSE
"""

    sample_tests = """
assert add(1, 2) == 3
assert add(-1, 5) == 4
print("All tests passed!")
"""

    task = f"""
You are given a buggy Python implementation and tests.

<code>
{sample_code}
</code>

<tests>
{sample_tests}
</tests>

Fix the code until ALL tests pass.
"""

    result = agent.run(task)
    print("\n=== FINAL RESULT ===")
    print(result["result"])
    print("Rounds:", result["rounds"])


if __name__ == "__main__":
    main()

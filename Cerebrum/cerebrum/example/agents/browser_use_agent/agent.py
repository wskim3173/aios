from typing import List, Dict, Any, Optional, Literal, Tuple, Union
import asyncio
import datetime
import io
import json
import os
import random
import re
import shutil
import time
import urllib.parse

from cerebrum.llm.apis import llm_chat, llm_chat_with_json_output, llm_chat_with_tool_call_output
from cerebrum.utils import _parse_json_output
from cerebrum.utils.browser import BaseBrowser, _reload_image

AVAILABLE_ACTIONS_PROMPT = """
1. `fill_input_id(identifier: Union[str, int], text: str)`: Fill an input
field (e.g. search box) with the given text and press Enter.
2. `click_id(identifier: Union[str, int])`: Click an element with the given ID.
3. `hover_id(identifier: Union[str, int])`: Hover over an element with the
given ID.
4. `download_file_id(identifier: Union[str, int])`: Download a file with the
given ID. It returns the path to the downloaded file. If the file is
successfully downloaded, you can stop the simulation and report the path to
the downloaded file for further processing.
5. `scroll_to_bottom()`: Scroll to the bottom of the page.
6. `scroll_to_top()`: Scroll to the top of the page.
7. `scroll_up()`: Scroll up the page. It is suitable when you want to see the
elements above the current viewport.
8. `scroll_down()`: Scroll down the page. It is suitable when you want to see
the elements below the current viewport. If the webpage does not change, It
means that the webpage has scrolled to the bottom.
9. `back()`: Navigate back to the previous page. This is useful when you want
to go back to the previous page, as current page is not useful.
10. `stop()`: Stop the action process, because the task is completed or failed
(impossible to find the answer). In this situation, you should provide your
answer in your output.
11. `get_url()`: Get the current URL of the current page.
12. `find_text_on_page(search_text: str)`: Find the next given text on the
current whole page, and scroll the page to the targeted text. It is equivalent
to pressing Ctrl + F and searching for the text, and is powerful when you want
to fast-check whether the current page contains some specific text.
13. `visit_page(url: str)`: Go to the specific url page.
14. `click_blank_area()`: Click a blank area of the page to unfocus the
current element. It is useful when you have clicked an element but it cannot
unfocus itself (e.g. Menu bar) to automatically render the updated webpage.
15. `ask_question_about_video(question: str)`: Ask a question about the
current webpage which contains video, e.g. youtube websites.
"""

ACTION_WITH_FEEDBACK_LIST = [
    'ask_question_about_video',
    'download_file_id',
    'find_text_on_page',
]

class BrowserUseAgent:
    r"""A class for browsing the web and interacting with web pages.

    This class provides methods for browsing the web and interacting with web
    pages.
    """

    def __init__(
        self,
        headless: bool = False,
        cache_dir: Optional[str] = None,
        channel: Literal["chrome", "msedge", "chromium"] = "chromium",
        history_window: int = 5,
    ):
        r"""Initialize the BrowserToolkit instance.

        Args:
            headless (bool): Whether to run the browser in headless mode.
            cache_dir (Union[str, None]): The directory to store cache files.
            channel (Literal["chrome", "msedge", "chromium"]): The browser
                channel to use. Must be one of "chrome", "msedge", or
                "chromium".
            history_window (int): The window size for storing the history of
                actions.
        """
        self.name = "browser_use_agent"
        self.browser = BaseBrowser(
            headless=headless, cache_dir=cache_dir, channel=channel
        )
        self.history_window = history_window
        self.history: list = []
        self.web_agent, self.planning_agent = self._initialize_agent()

    def _reset(self):
        self.web_agent.reset()
        self.planning_agent.reset()
        self.history = []
        os.makedirs(self.browser.cache_dir, exist_ok=True)

    def _initialize_agent(self):
        r"""Initialize the agent."""
        class WebAgent:
            def __init__(self, system_prompt: str):
                self.name = "web_agent"
                self.system_prompt = system_prompt
                self.messages = [
                    {"role": "system", "content": system_prompt}
                ]
            
            def step(self, message, response_format=None, tools=None):
                self.messages.append(message)
                llms = [
                    {
                        "name": "gpt-4o-mini",
                        "backend": "openai",
                    }
                ]
                if tools:
                    response = llm_chat_with_tool_call_output(
                        agent_name=self.name,
                        messages=self.messages,
                        llms=llms,
                        tools=tools
                    )["response"]
                elif response_format:
                    response = llm_chat_with_json_output(
                        agent_name=self.name,
                        messages=self.messages,
                        llms=llms,
                        response_format=response_format
                    )["response"]
                else:
                    response = llm_chat(
                        agent_name=self.name,
                        messages=self.messages,
                        llms=llms,
                    )["response"]
                return response
            
            def reset(self):
                self.messages = [
                    {"role": "system", "content": self.system_prompt}
                ]

        class PlanningAgent:
            def __init__(self, system_prompt: str):
                self.name = "planning_agent"
                self.system_prompt = system_prompt
                self.messages = [
                    {"role": "system", "content": system_prompt}
                ]
            
            def step(self, message, response_format=None, tools=None):
                self.messages.append(message)
                llms = [
                    {
                        "name": "gpt-4o-mini",
                        "backend": "openai",
                    }
                ]
                if tools:
                    response = llm_chat_with_tool_call_output(
                        agent_name=self.name,
                        messages=self.messages,
                        llms=llms,
                        tools=tools
                    )["response"]
                elif response_format:
                    response = llm_chat_with_json_output(
                        agent_name=self.name,
                        messages=self.messages,
                        llms=llms,
                        response_format=response_format
                    )["response"]
                else:
                    response = llm_chat(
                        agent_name=self.name,
                        messages=self.messages,
                        llms=llms,
                    )["response"]
                return response
            
            def reset(self):
                self.messages = [
                    {"role": "system", "content": self.system_prompt}
                ]
        
        system_prompt = """
You are a helpful web agent that can assist users in browsing the web.
Given a high-level task, you can leverage predefined browser tools to help
users achieve their goals.
        """

        web_agent = WebAgent(system_prompt)

        planning_system_prompt = """
You are a helpful planning agent that can assist users in planning complex
tasks which need multi-step browser interaction.
        """

        planning_agent = PlanningAgent(planning_system_prompt)

        return web_agent, planning_agent
    
    def convert_message(self, message, img=None):
        import base64
        from io import BytesIO

        if img is not None:
            # Convert PIL Image to base64
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": message,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    }
                ]
            }
        else:
            return {
                "role": "user",
                "content": message
            }

    def _observe(
        self, task_prompt: str, detailed_plan: Optional[str] = None
    ) -> Tuple[str, str, str]:
        r"""Let agent observe the current environment, and get the next action."""

        detailed_plan_prompt = ""

        if detailed_plan is not None:
            detailed_plan_prompt = f"""
Here is a plan about how to solve the task step-by-step which you must follow:
<detailed_plan>{detailed_plan}<detailed_plan>
        """

        observe_prompt = f"""
Please act as a web agent to help me complete the following high-level task:
<task>{task_prompt}</task>
Now, I have made screenshot (only the current viewport, not the full webpage)
based on the current browser state, and marked interactive elements in the
webpage.
Please carefully examine the requirements of the task, and current state of
the browser, and provide the next appropriate action to take.

{detailed_plan_prompt}

Here are the current available browser functions you can use:
{AVAILABLE_ACTIONS_PROMPT}

Here are the latest {self.history_window} trajectory (at most) you have taken:
<history>
{self.history[-self.history_window:]}
</history>

Your output should be in json format, including the following fields:
- `observation`: The detailed image description about the current viewport. Do
not over-confident about the correctness of the history actions. You should
always check the current viewport to make sure the correctness of the next
action.
- `reasoning`: The reasoning about the next action you want to take, and the
possible obstacles you may encounter, and how to solve them. Do not forget to
check the history actions to avoid the same mistakes.
- `action_code`: The action code you want to take. It is only one step action
code, without any other texts (such as annotation)

Here is two example of the output:
```json
{{
    "observation": [IMAGE_DESCRIPTION],
    "reasoning": [YOUR_REASONING],
    "action_code": "fill_input_id([ID], [TEXT])"
}}

{{
    "observation":  "The current page is a CAPTCHA verification page on Amazon. It asks the user to ..",
    "reasoning": "To proceed with the task of searching for products, I need to complete..",
    "action_code": "fill_input_id(3, 'AUXPMR')"
}}

Here are some tips for you:
- Never forget the overall question: **{task_prompt}**
- Maybe after a certain operation (e.g. click_id), the page content has not
changed. You can check whether the action step is successful by looking at the
`success` of the action step in the history. If successful, it means that the
page content is indeed the same after the click. You need to try other methods.
- If using one way to solve the problem is not successful, try other ways.
Make sure your provided ID is correct!
- Some cases are very complex and need to be achieve by an iterative process.
You can use the `back()` function to go back to the previous page to try other
methods.
- There are many links on the page, which may be useful for solving the
problem. You can use the `click_id()` function to click on the link to see if
it is useful.
- Always keep in mind that your action must be based on the ID shown in the
current image or viewport, not the ID shown in the history.
- Do not use `stop()` lightly. Always remind yourself that the image only
shows a part of the full page. If you cannot find the answer, try to use
functions like `scroll_up()` and `scroll_down()` to check the full content of
the webpage before doing anything else, because the answer or next key step
may be hidden in the content below.
- If the webpage needs human verification, you must avoid processing it.
Please use `back()` to go back to the previous page, and try other ways.
- If you have tried everything and still cannot resolve the issue, please stop
the simulation, and report issues you have encountered.
- Check the history actions carefully, detect whether you have repeatedly made
the same actions or not.
- When dealing with wikipedia revision history related tasks, you need to
think about the solution flexibly. First, adjust the browsing history
displayed on a single page to the maximum, and then make use of the
find_text_on_page function. This is extremely useful which can quickly locate
the text you want to find and skip massive amount of useless information.
- Flexibly use interactive elements like slide down selection bar to filter
out the information you need. Sometimes they are extremely useful.
"""

        som_screenshot, _ = self.browser.get_som_screenshot(save_image=True)
        img = _reload_image(som_screenshot)
        message = self.convert_message(observe_prompt, img)
        self.web_agent.reset()
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "react",
                "schema": {
                    "type": "object",
                    "properties": {
                        "observation": {"type": "string"},
                        "reasoning": {"type": "string"},
                        "action_code": {"type": "string"}
                    },
                    "required": ["observation", "reasoning", "action_code"]
                },
                "strict": True
            }
        }
        
        response = self.web_agent.step(message, response_format=response_format)["response_message"]
        response = _parse_json_output(response)
        observation_result: str = response.get("observation", "")
        reasoning_result: str = response.get("reasoning", "")
        action_code: str = response.get("action_code", "")

        if action_code and "(" in action_code and ")" not in action_code:
            action_match = re.search(
                r'"action_code"\s*:\s*[`"]([^`"]*\([^)]*\))[`"]', response
            )
            if action_match:
                action_code = action_match.group(1)
            else:
                print(
                    f"Incomplete action_code detected: {action_code}"
                )
                if action_code.startswith("fill_input_id("):
                    parts = action_code.split(",", 1)
                    if len(parts) > 1:
                        id_part = (
                            parts[0].replace("fill_input_id(", "").strip()
                        )
                        action_code = f"fill_input_id({id_part}, 'Please fill the text here.')"

        action_code = action_code.replace("`", "").strip()

        return observation_result, reasoning_result, action_code

    def _act(self, action_code: str) -> Tuple[bool, str]:
        r"""Let agent act based on the given action code.
        Args:
            action_code (str): The action code to act.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether
                the action was successful, and the information to be returned.
        """

        def _check_if_with_feedback(action_code: str) -> bool:
            r"""Check if the action code needs feedback."""

            for action_with_feedback in ACTION_WITH_FEEDBACK_LIST:
                if action_with_feedback in action_code:
                    return True

            return False

        def _fix_action_code(action_code: str) -> str:
            r"""Fix potential missing quotes in action code"""

            match = re.match(r'(\w+)\((.*)\)', action_code)
            if not match:
                return action_code

            func_name, args_str = match.groups()

            args = []
            current_arg = ""
            in_quotes = False
            quote_char = None

            for char in args_str:
                if char in ['"', "'"]:
                    if not in_quotes:
                        in_quotes = True
                        quote_char = char
                        current_arg += char
                    elif char == quote_char:
                        in_quotes = False
                        quote_char = None
                        current_arg += char
                    else:
                        current_arg += char
                elif char == ',' and not in_quotes:
                    args.append(current_arg.strip())
                    current_arg = ""
                else:
                    current_arg += char

            if current_arg:
                args.append(current_arg.strip())

            fixed_args = []
            for arg in args:
                if (
                    (arg.startswith('"') and arg.endswith('"'))
                    or (arg.startswith("'") and arg.endswith("'"))
                    or re.match(r'^-?\d+(\.\d+)?$', arg)
                    or re.match(r'^-?\d+\.?\d*[eE][-+]?\d+$', arg)
                    or re.match(r'^0[xX][0-9a-fA-F]+$', arg)
                ):
                    fixed_args.append(arg)
                else:
                    fixed_args.append(f"'{arg}'")

            return f"{func_name}({', '.join(fixed_args)})"

        action_code = _fix_action_code(action_code)
        prefix = "self.browser."
        code = f"{prefix}{action_code}"

        try:
            if _check_if_with_feedback(action_code):
                # execute code, and get the executed result
                result = eval(code)
                time.sleep(1)
                return True, result
            else:
                exec(code)
                time.sleep(1)
                return True, "Action was successful."

        except Exception as e:
            time.sleep(1)
            return (
                False,
                f"Error while executing the action {action_code}: {e}. "
                f"If timeout, please recheck whether you have provided the "
                f"correct identifier.",
            )

    def _get_final_answer(self, task_prompt: str) -> str:
        r"""Get the final answer based on the task prompt and current browser state.
        It is used when the agent thinks that the task can be completed without any further action, and answer can be directly found in the current viewport.
        """

        prompt = f"""
We are solving a complex web task which needs multi-step browser interaction. After the multi-step observation, reasoning and acting with web browser, we think that the task is currently solved.
Here are all trajectory we have taken:
<history>{self.history}</history>
Please find the final answer, or give valuable insights and founds (e.g. if previous actions contain downloading files, your output should include the path of the downloaded file) about the overall task: <task>{task_prompt}</task>
        """

        message = {
            "role": "user",
            "content": prompt
        }

        response = self.web_agent.step(message)
        return response

    def _make_reflection(self, task_prompt: str) -> str:
        r"""Make a reflection about the current state and the task prompt."""

        reflection_prompt = f"""
Now we are working on a complex task that requires multi-step browser interaction. The task is: <task>{task_prompt}</task>
To achieve this goal, we have made a series of observations, reasonings, and actions. We have also made a reflection on previous states.

Here are the global available browser functions we can use:
{AVAILABLE_ACTIONS_PROMPT}

Here are the latest {self.history_window} trajectory (at most) we have taken:
<history>{self.history[-self.history_window:]}</history>

The image provided is the current state of the browser, where we have marked interactive elements. 
Please carefully examine the requirements of the task, and the current state of the browser, and then make reflections on the previous steps, thinking about whether they are helpful or not, and why, offering detailed feedback and suggestions for the next steps.
Your output should be in json format, including the following fields:
- `reflection`: The reflection about the previous steps, thinking about whether they are helpful or not, and why, offering detailed feedback.
- `suggestion`: The suggestion for the next steps, offering detailed suggestions, including the common solutions to the overall task based on the current state of the browser.
        """
        som_image, _ = self.browser.get_som_screenshot()
        img = _reload_image(som_image)

        message = self.convert_message(reflection_prompt, img)

        response = self.web_agent.step(message)

        return response

    def _task_planning(self, task_prompt: str, start_url: str) -> str:
        r"""Plan the task based on the given task prompt."""

        planning_prompt = f"""
<task>{task_prompt}</task>
According to the problem above, if we use browser interaction, what is the general process of the interaction after visiting the webpage `{start_url}`? 

Please note that it can be viewed as Partially Observable MDP. Do not over-confident about your plan.
Please first restate the task in detail, and then provide a detailed plan to solve the task.
"""

        message = self.convert_message(planning_prompt)

        response = self.planning_agent.step(message)
        return response["response_message"]

    def _task_replanning(
        self, task_prompt: str, detailed_plan: str
    ) -> Tuple[bool, str]:
        r"""Replan the task based on the given task prompt.

        Args:
            task_prompt (str): The original task prompt.
            detailed_plan (str): The detailed plan to replan.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating whether the task needs to be replanned, and the replanned schema.
        """

        replanning_prompt = f"""
We are using browser interaction to solve a complex task which needs multi-step actions.
Here are the overall task:
<overall_task>{task_prompt}</overall_task>

In order to solve the task, we made a detailed plan previously. Here is the detailed plan:
<detailed plan>{detailed_plan}</detailed plan>

According to the task above, we have made a series of observations, reasonings, and actions. Here are the latest {self.history_window} trajectory (at most) we have taken:
<history>{self.history[-self.history_window:]}</history>

However, the task is not completed yet. As the task is partially observable, we may need to replan the task based on the current state of the browser if necessary.
Now please carefully examine the current task planning schema, and our history actions, and then judge whether the task needs to be fundamentally replanned. If so, please provide a detailed replanned schema (including the restated overall task).

Your output should be in json format, including the following fields:
- `if_need_replan`: bool, A boolean value indicating whether the task needs to be fundamentally replanned.
- `replanned_schema`: str, The replanned schema for the task, which should not be changed too much compared with the original one. If the task does not need to be replanned, the value should be an empty string. 
"""
        self.planning_agent.reset()
        
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "react",
                "schema": {
                    "type": "object",
                    "properties": {
                        "if_need_replan": {"type": "boolean"},
                        "replanned_schema": {"type": "string"}
                    },
                    "required": ["if_need_replan", "replanned_schema"]
                }
            }
        }
        
        response = self.planning_agent.step(replanning_prompt, response_format=response_format)
        
        resp_dict = _parse_json_output(response)
        if_need_replan = resp_dict.get("if_need_replan", False)
        replanned_schema = resp_dict.get("replanned_schema", "")

        if if_need_replan:
            return True, replanned_schema
        else:
            return False, replanned_schema

    def run(
        self, task_input: str, start_url: str, round_limit: int = 50
    ) -> str:
        r"""A powerful toolkit which can simulate the browser interaction to solve the task which needs multi-step actions.

        Args:
            task_prompt (str): The task prompt to solve.
            start_url (str): The start URL to visit.
            round_limit (int): The round limit to solve the task.
                (default: :obj:`12`).

        Returns:    
            str: The simulation result to the task.
        """

        self._reset()
        task_completed = False
        
        detailed_plan = self._task_planning(task_input, start_url)
        print(f"Detailed plan: {detailed_plan}")

        self.browser.init()
        self.browser.visit_page(start_url)
        
        rounds = 0

        while rounds < round_limit:
            observation, reasoning, action_code = self._observe(
                task_input, detailed_plan
            )
            print(f"Observation: {observation}")
            print(f"Reasoning: {reasoning}")
            print(f"Action code: {action_code}")

            if "stop" in action_code:
                task_completed = True
                trajectory_info = {
                    "round": rounds,
                    "observation": observation,
                    "thought": reasoning,
                    "action": action_code,
                    "action_if_success": True,
                    "info": None,
                    "current_url": self.browser.get_url(),
                }
                self.history.append(trajectory_info)
                break

            else:
                success, info = self._act(action_code)
                if not success:
                    print(f"Error while executing the action: {info}")

                trajectory_info = {
                    "round": rounds,
                    "observation": observation,
                    "thought": reasoning,
                    "action": action_code,
                    "action_if_success": success,
                    "info": info,
                    "current_url": self.browser.get_url(),
                }
                self.history.append(trajectory_info)
                    
            rounds += 1

        if not task_completed:
            simulation_result = f"""
                The task is not completed within the round limit. Please check the last round {self.history_window} information to see if there is any useful information:
                <history>{self.history[-self.history_window:]}</history>
            """

        else:
            simulation_result = self._get_final_answer(task_input)

        self.browser.close()
        return {
            "agent_name": self.name,
            "rounds": rounds,
            "result": simulation_result
        }


def main():
    agent = BrowserUseAgent()
    task_input = "What is the densest material on the moon?"
    start_url = "https://www.wikipedia.org"
    agent.run(task_input, start_url)

if __name__ == "__main__":
    main()
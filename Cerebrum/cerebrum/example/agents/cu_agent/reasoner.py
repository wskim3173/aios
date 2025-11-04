import copy
from typing import List, Dict, Tuple, Optional
from cerebrum.llm.apis import llm_chat_with_json_output
from cerebrum.utils.utils import _parse_json_output

from pydantic import BaseModel

class ActionCode(BaseModel):
    action_type: str
    parameters: Dict

class ReasoningOutput(BaseModel):
    thought: str
    action: str
    action_code: ActionCode
    

class Reasoner:
    def __init__(self, mcp_client, llms: List[Dict], history_window: int = 15, round_limit: int = 50):
        self.llms = llms
        self.name = "reasoner"
        self.mcp_client = mcp_client
        self.history_window = history_window
        self.round_limit = round_limit
#         self.action_space = """
# - move_mouse(x: int, y: int)
# - click(x: int = None, y: int = None, num_clicks: int = None)
# - right_click(x: int = None, y: int = None)
# - double_click(x: int = None, y: int = None)
# - typing(text: str)
# - drag_to(x: int, y: int)
# - scroll(dx: int, dy: int)
# - press(key: str)
# - hotkey(keys: List[str])
# - wait(seconds: int = None)
# - done
# - fail 
# """
        # self.screen_size = (1920, 1080)
        self.screen_size = (1440, 900)

        self.action_tips = f"""
## Rules
- Keep the mission in sight<br>Write the task objective at the top of your internal scratchpad and re-evaluate it after every observation step. 
- Recognize which app(s) you need to open to complete the task, if your task just involves single application, do not open other applications! 
- If you need to save/open/download a file to a specific location, you need to first analyze and identify the absolute path of the target location from user intention. 
- Before scrolling, remember to click the scroll bar to make sure scrolling is enabled first and then scroll. Sometimes you can click the position above or below the scroll bar to make scrolling work.
- Finish or bail promptly<br>If you are **certain** the job is done, send the `done` action; if you reach *{self.round_limit - self.round_limit // 5}* rounds and still have several unmet milestones, emit `fail` next turn.
- Batch work in the terminal<br>List, search, rename, or grep files with shell commands rather than clicking each item. CLI is measurably faster when the path is known.
- Use hotkeys (`["ctrl","a"]`, `["ctrl","c"]`, `["ctrl","v"]`, `["alt","tab"]`, etc.) when you need to explicitly select all, copy, paste, switch window, etc, before resorting to mouse moves.
- Leverage double- / right-click<br>Opening files or context menus often succeeds with a double- or right-click when a plain click fails; try these before re-locating the element.
- Never retry the same move > 2 times<br>After two identical failures, switch tactic (e.g., switch click to double-click, or scroll then click).
- Correct for imprecise coords<br>If hits miss, nudge the click point ±3 % of *{self.screen_size}* after visually confirming the element’s real location.
- Scroll first, then interact<br>If the element is off-screen, scroll until visible before clicking; many libraries throw “element not in viewport” otherwise.
- Ensure focus before typing<br>Double-click or `Tab` into the field, verify the caret is blinking, then type. This prevents focus-loss bugs.
- When you need to type text and then to search for something, do not repeat typing the text! The correct order is just type the text and press enter/ use click button to search, double check whether the field you need to type text has already been filled, 
- Close or hide noise windows<br>Shut any pop-ups or tabs unrelated to the task to stop mis-focus and selector clashes. 
- Checkpoint after every 5 rounds<br>Compare completed milestones vs. remaining rounds; adjust plan instead of blindly continuing.
- When you think you have finished the task, just output the `done` action. No need to relaunch or close the tab/browser/application or verify that by yourself, the tabs need to be open for the verification. 
- When you need to install extension/package from your local device, you can first check whether the extension/package can be dragged to the browser for installation. 
- When installing extensions, try to use extension tab natively in the application/browser. 

## Extra safeguards against common mistakes
- Keep a small counter per element/action. If an element receives over two consecutive identical clicks or a text box receives the same string more than twice, choose another method (scroll, re-focus, keyboard shortcut, or alternate selector. 
- When you need to adjust the value of an element, such as size, volume, you can first look at whether you can directly type the new value in the field, if not, you can try passing the number of clicks to action `click` to adjust the value.
- Use terminal to download apps such as sudo apt install, if install from terminal can not work, use the application store to install the app. 
- When you mistakenly type too much text in the search bar, you can first use hotkey `["ctrl","a"]` and then press `backspace` to delete the text. 
- Before any `type` or `hotkey`, make sure the focus is on the correct element, read the active-window title and ensure it matches the expected app/tab. Abort typing if it differs. 
- When multiple tabs/windows exist, always bring the correct one to foreground. Close rogue tabs right away. 
"""

    def reset(self):
        self.history = []
        
    async def construct_messages(
        self,
        base64_screenshot: Optional[str] = None,
        action_space: Optional[str] = None,
        observation_prompt: Optional[str] = None
    ) -> List[Dict]:
        messages = [
            {
                "role": "system",
                "content": f"""
You are a reasoner that can analyze the current state, previous actions, and task progress to determine what needs to be done next. 
The user_name is "user" and the password is "password", remember to use them when requiring sudo permission.
You can only take the actions in the following action space:
<action_space>
{action_space}
</action_space>
And here are some tips for you:
<action_tips>
{self.action_tips}
</action_tips>
"""
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": observation_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_screenshot}"}
                    }
                ]
            }
        ]
        return messages
        
    async def reasoning(
        self, 
        base64_screenshot: Optional[str] = None,
        elements_info: Optional[str] = None,
        plan: Optional[str] = None,
        history: Optional[List[Dict]] = None,
        task_input: Optional[str] = None,
    ) -> Tuple[str, str, Dict]:
        """Analyze the current state and determine next action.
        
        Args:
            base64_screenshot (Optional[str]): The current screenshot
            elements_info (Optional[str]): Information about UI elements
            plan (Optional[str]): The current task plan
            history (Optional[List[Dict]]): Previous actions and observations
            
        Returns:
            Tuple[str, str, Dict]: (thought, action, action_code)
        """
        response_format = ReasoningOutput.model_json_schema()
        observation_prompt = f"""
Keep your overall task in mind:
<task>
{task_input}
</task>
Based on the current screenshot and your plan
<plan>
{plan}
</plan>
and your previous thought and action histories <history>{history[-self.history_window:] if history else []}</history>
examine the current progress of the task. 

Given the elements in the current screenshot as below:
<elements_info>
{elements_info}
</elements_info>
and provide the next appropriate action to take. 

Output the following fields in JSON format:
- `thought`: You thought of the current environment, the status of the previous action, the progress of the task. 
- `action`: The specific action you want to take. 

Make sure that the thought, action and action_code must be consistent with each other, 
which means your action should be based on your thought and the action_code should be the action you want to take. 

Here is an example of the output:
```json
{{
    "thought": "It currently shows the login page of the website, it suggests that the previous action of opening the browser is successful. ",
    "action": "I will click the login button at the right-top corner of the page.",
    "action_code": {{
        "action_type": "click",
        "parameters": {{
            "x": 100,
            "y": 100,
            "num_clicks": 1
        }}
    }}
}}
{{
    "thought": "The current volume is 50, I need to increase it to 80. ",
    "action": "I will click the volume up button 3 times. ",
    "action_code": {{
        "action_type": "click",
        "parameters": {{
            "x": 100,
            "y": 100,
            "num_clicks": 3
        }}
    }}
}}
{{
    "thought": "I have successfully renamed the file and no further action is needed. ",
    "action": "No more action is needed. ",
    "action_code": {{
        "action_type": "done",
        "parameters": {{}}
    }}
}}
{{
    "thought": "I have already typed the password in the terminal, and I just need to wait for a few seconds. ",
    "action": "Wait for a few seconds. ",
    "action_code": {{
        "action_type": "wait",
        "parameters": {{
            "seconds": 3
        }}
    }}
}}
{{
    "thought": "I only have 3 rounds left, but I have not found the target file to search and modify texts in this file. ",
    "action": "Give up the task. ",
    "action_code": {{
        "action_type": "fail",
        "parameters": {{}}
    }}
}}
```
"""     
        action_space = await self.mcp_client.get_tool_schemas()
        
        messages = await self.construct_messages(
            base64_screenshot=base64_screenshot,
            action_space=action_space,
            observation_prompt=observation_prompt
        )
        
        response = llm_chat_with_json_output(
            agent_name=self.name,
            messages=messages,
            llms=self.llms,
            response_format=response_format
        )["response"]["response_message"]
        
        response_dict = _parse_json_output(response)
        thought: str = response_dict.get("thought", "")
        action: str = response_dict.get("action", "")
        action_code: dict = response_dict.get("action_code", {})
        
        return thought, action, action_code
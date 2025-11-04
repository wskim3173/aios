import copy
from typing import List, Dict
from cerebrum.llm.apis import llm_chat_with_json_output
from cerebrum.utils.utils import _parse_json_output
from pydantic import BaseModel

from typing import Tuple, List


class PerceiverOutput(BaseModel):
    elements_usage: List[str]
    
class Perceiver:
    def __init__(self, llms: List[Dict]):
        self.llms = llms
        self.name = "perceiver"
        self.system_prompt = f"""
You are a perceiver that can help perceive the elements in the screenshot and understand the usage of the elements.
"""
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
    def reset(self):
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
    def perceive(self, base64_screenshot: str, active_elements: List[Dict]) -> Dict:
        """Perceive the elements in the screenshot and understand the usage of the elements.
        
        Args:
            base64_screenshot (str): The current screenshot
            active_elements (List[Dict]): Information about available UI elements
            
        Returns:
            Dict: The perceived elements with the usage of the elements
        """
        
        current_messages = copy.deepcopy(self.messages)
        user_message = f"""
        Based on the current screenshot and provided active elements, please provide the usage for each element based on your understanding of the screenshot:
Here are the active elements in the current screenshot:
<elements>
{active_elements}
</elements>
Output the usage of each element in the JSON format, the key of the output must be "elements_usage" and its value must be a list of strings, 
when describing the usage of the element, be concise and to the point.

Here is an example of the output:
```json
{{
    "elements_usage": [
        "Used to open the settings menu",
        "Used to close the current window"
        ...
    ]
}}
```
"""
        current_messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_screenshot}"}}
            ]
        })
        
        response = llm_chat_with_json_output(
            agent_name=self.name,
            messages=current_messages,
            llms=self.llms,
            response_format=PerceiverOutput.model_json_schema()
        )["response"]["response_message"]
        
        # breakpoint()
        
        response_dict = _parse_json_output(response)
        elements_usage = response_dict.get("elements_usage", [])
        # breakpoint()
        if len(elements_usage) != len(active_elements):
            elements_usage = []
        elements_info = self.construct_elements_info(active_elements, elements_usage)
        return elements_info
    
    def construct_elements_info(self, active_elements: List[Dict], elements_usage: List[str]) -> str:
        elements_info = "Here are the elements in the current screenshot, with format as - Usage; Location\n"
        for i, element in enumerate(active_elements):
            element_type = element["type"]
            try:
                element_usage = elements_usage[i]
            except:
                element_usage = element["description"]
                
            raw_loc = element["location"]
            x = int(raw_loc[0] + raw_loc[2] / 2)
            y = int(raw_loc[1] + raw_loc[3] / 2)
            element_loc = f"{{x: {x}, y: {y}}}"
            elements_info += f"- {element_usage}; {element_loc}\n"
        return elements_info

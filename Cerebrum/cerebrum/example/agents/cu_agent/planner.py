import copy
from typing import List, Dict
from cerebrum.llm.apis import llm_chat

class Planner:
    def __init__(self, llms: List[Dict]):
        self.llms = llms
        self.name = "planner"
        self.messages = []
        self.system_prompt = """
You are a planning agent that can help break down complex computer tasks into actionable steps.
Given a high-level task, you can generate a detailed plan of actions needed to complete it.
"""
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
    def reset(self):
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
    def plan(self, task_input: str) -> str:
        """Generate a detailed plan for completing the given task.
        
        Args:
            task_input (str): The high-level task description
            
        Returns:
            str: A detailed plan with milestones and steps
        """
        current_messages = copy.deepcopy(self.messages)
        
        user_message = f"""
Generate a detailed plan of the actions you would like to take to complete the task, including several milestones that can help you examine the current progress of the task.
Task: {task_input}
"""
        current_messages.append({
            "role": "user",
            "content": user_message
        })
        
        response = llm_chat(
            agent_name=self.name,
            messages=current_messages,
            llms=self.llms,
        )["response"]["response_message"]
        
        return response 
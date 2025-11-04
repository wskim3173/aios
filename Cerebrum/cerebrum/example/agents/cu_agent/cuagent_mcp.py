import datetime
import io
import json
import os
import random
import re
import shutil
import time
import urllib.parse
import ast

from PIL import Image
import io
import matplotlib.pyplot as plt

from cerebrum.llm.apis import llm_chat, llm_chat_with_json_output, llm_chat_with_tool_call_output
from cerebrum.utils.utils import _parse_json_output

from cerebrum.utils.communication import get_mcp_server_path, aios_kernel_url

mcp_server_path = get_mcp_server_path(aios_kernel_url)

import base64

import xml.etree.ElementTree as ET

from .mcp_client import MCPClient
from .planner import Planner
from .perceiver import Perceiver
from .reasoner import Reasoner
from .actor import Actor

from typing import Tuple, List, Dict, Union

import asyncio

import tiktoken

from .accessibility_tree_wrap.heuristic_retrieve import get_ubuntu_interactive_leaf_elements
import copy

class CUAgent:
    def __init__(
        self,
        screen_size: Tuple[int, int] = (1920, 1080),
        history_window: int = 15,
        mcp_client: MCPClient = None,
        task_config: dict = None,
    ):
        self.screen_size = screen_size
        self.history_window = history_window
        self.task_config = task_config
        self.cache_dir = os.path.join(os.path.dirname(__file__), "cache", self.task_config["domain"], self.task_config["id"])
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            
        self.orchestrator_llms = [
            {
                "name": "gpt-4o",
                "backend": "openai",
            }
        ]
            
        # Initialize modules
        self.planner_llms = [
            {
                "name": "gpt-4o",
                "backend": "openai",
            }
        ]
        
        self.perceiver_llms = [
            {
                "name": "gpt-4o",
                "backend": "openai",
            }
        ]
        
        self.reasoner_llms = [
            {
                "name": "gpt-4o",
                "backend": "openai",
            }
        ]
        
        self.planner = Planner(self.planner_llms)
        self.perceiver = Perceiver(self.perceiver_llms)
        self.reasoner = Reasoner(
            mcp_client=mcp_client,
            llms=self.reasoner_llms,
            history_window=history_window
        )
        self.actor = Actor(mcp_client)
        
        # State tracking
        self.history = []
        self.action_history = []
        self.plan = None
        
    def save_screenshot(self, bytes_obj, tagged=False):
        if isinstance(bytes_obj, bytes):
            bytes_io = io.BytesIO(bytes_obj)
        elif isinstance(bytes_obj, str):
            bytes_obj = self.bytes_literal_to_bytesio(bytes_obj)
            bytes_io = io.BytesIO(bytes_obj)
        else:
            raise ValueError("type of bytes_obj is not supported, only bytes or str is supported")

        img = Image.open(bytes_io)
            
        os.makedirs(os.path.join(self.cache_dir, "screenshots"), exist_ok=True)
        
        if tagged:
            img_path = os.path.join(self.cache_dir, "screenshots", f"screenshot_{time.time()}_tagged.png")
        else:
            img_path = os.path.join(self.cache_dir, "screenshots", f"screenshot_{time.time()}.png")
        img.save(img_path)
        
    def encode_screenshot(self, screenshot: Union[bytes, str]) -> str:
        if isinstance(screenshot, bytes):
            return base64.b64encode(screenshot).decode("utf-8")
        elif isinstance(screenshot, str):
            bytes_obj = self.bytes_literal_to_bytesio(screenshot)
            return base64.b64encode(bytes_obj).decode("utf-8")
        else:
            raise ValueError("type of screenshot is not supported, only bytes or str is supported")
    
    def bytes_literal_to_bytesio(self, bytes_literal_str):
        bytes_obj = ast.literal_eval(bytes_literal_str)
        
        if not isinstance(bytes_obj, bytes):
            raise ValueError("not a valid bytes literal")
        
        return bytes_obj
    
    def _construct_elements_info(self, active_elements: list) -> str:
        elements_info = "Here are the elements in the current screenshot, with format as - Tppe; Usage; Location\n"
        for i, element in enumerate(active_elements):
            element_type = element["type"]
            element_usage = element["description"]
                
            raw_loc = element["location"]
            x = int(raw_loc[0] + raw_loc[2] / 2)
            y = int(raw_loc[1] + raw_loc[3] / 2)
            element_loc = f"{{x: {x}, y: {y}}}"
            elements_info += f"- {element_type}; {element_usage}; {element_loc}\n"
        return elements_info
    
    def save_trajectories(self, trajectories: dict):
        with open(os.path.join(self.cache_dir, "trajectories.json"), "w") as f:
            json.dump(trajectories, f, indent=4)
    
    async def run(
        self, task_input: str, round_limit: int = 50
    ) -> str:
        """Run the agent to complete a task.
        
        Args:
            task_input (str): The task description
            round_limit (int): Maximum number of rounds to run
            
        Returns:
            str: Result of the task execution
        """
        # Reset state
        self.history = []
        self.action_history = []
        self.planner.reset()
        self.perceiver.reset()
        self.reasoner.reset()
        
        # Reset VM
        # await self.actor.mcp_client.call_tool("reset_vm", {"task_config": self.task_config})
        
        rounds = 0
        
        # Start recording
        # await self.actor.start_recording()
        
        # Generate plan
        self.plan = self.planner.plan(task_input)
        
        while rounds < round_limit:
            # Get current state
            screenshot = await self.actor.get_screenshot()
            self.save_screenshot(screenshot, tagged=False)
            
            accessibility_tree = await self.actor.get_accessibility_tree()
            
            base64_screenshot = self.encode_screenshot(screenshot)
            active_elements = get_ubuntu_interactive_leaf_elements(accessibility_tree)
            # elements_info = self.perceiver.perceive(base64_screenshot, active_elements)
            elements_info = self._construct_elements_info(active_elements)
            # Perceive and decide next action
            thought, action, action_code = await self.reasoner.reasoning(
                task_input=task_input,
                base64_screenshot=base64_screenshot,
                elements_info=elements_info,
                plan=self.plan,
                history=self.history,
            )
            
            print(f"Thought: {thought}")
            print(f"Action: {action}")
            print(f"Action Code: {action_code}")
            
            action_type = action_code.get("action_type", "")
            
            # Handle special actions
            if "DONE" in action_type or "done" in action_type:
                trajectory_info = {
                    "round": rounds,
                    "thought": thought,
                    "action": action,
                    "info": None,
                }
                self.history.append(trajectory_info)
                self.action_history.append(action_code)
                break
            
            elif "FAIL" in action_type or "fail" in action_type:
                trajectory_info = {
                    "round": rounds,
                    "thought": thought,
                    "action": action,
                }
                self.history.append(trajectory_info)
                self.action_history.append(action_code)
                break
            
            # Execute action
            # breakpoint()
            # action_result = await self.actor.act(action_code)
            await self.actor.act(action_code)
            
            trajectory_info = {
                "round": rounds,
                "thought": thought,
                "action": action,
            }
            self.history.append(trajectory_info)
            self.action_history.append(action_code)
            
            rounds += 1
            
            time.sleep(1)
            
        # Stop recording
        # await self.actor.stop_recording(
        #     self.cache_dir + time.strftime("/recording-%Y%m%d-%H%M%S.mp4")
        # )
        
        # Evaluate performance
        # score = await self.actor.evaluate(self.action_history)
        score = 0
        
        # Save trajectories
        trajectories = {
            "task": self.task_config["instruction"],
            "reasoning": self.history,
            "action": self.action_history,
            "reward": score
        }
        self.save_trajectories(trajectories)
        
        summarization_prompt = f"""
        You are a helpful assistant that summarizes the final result of the task.
        Here is the task: {self.task_config["instruction"]} and here are the actions you have taken: {self.history}
        Please summarize the final result of the task.
        """
        final_result = llm_chat(
            agent_name="summarizer",
            llms=self.orchestrator_llms,
            messages=[
                {"role": "user", "content": summarization_prompt},
            ]
        )["response"]["response_message"]
        
        return {
            "agent_name": "cu_agent",
            "rounds": rounds,
            "result": final_result,
            # "score": score
        }

async def run_cu_agent(task_config: dict, mcp_server_path: str):
    mcp_client = MCPClient()
    try:
        await mcp_client.connect(mcp_server_path)
        agent = CUAgent(
            mcp_client=mcp_client,
            task_config=task_config
        )
        response = await agent.run(task_config["instruction"])
        result = response["result"]
        print(result)
            
    finally:
        await mcp_client.close()
        
if __name__ == "__main__":
    import sys
    
    task_input = sys.argv[1]
    task_config = {
        "instruction": task_input,
        "domain": "chrome", 
        "task_id": 0,
    }
    asyncio.run(run_cu_agent(task_config, mcp_server_path))
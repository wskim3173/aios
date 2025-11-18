from cerebrum.llm.apis import llm_chat, llm_chat_with_json_output, llm_chat_with_tool_call_output

from litellm import completion

from benchmarks.utils import get_parser

from datasets import load_dataset

from dotenv import load_dotenv

from typing import List, Dict, Any

import asyncio

import json

import uuid

from cerebrum.example.agents.browser_use_agent.agent import BrowserUseAgent
from cerebrum.example.agents.code_executor.agent import CodeExecutor
from cerebrum.example.agents.calculator_agent.agent import CalculatorAgent

from cerebrum.utils import _parse_json_output

from cerebrum.interface import AutoTool

load_dotenv()

class ReActAgent:
    def __init__(self, on_aios: bool = True):
        self.agent_name = "react"
        self.on_aios = on_aios
        self.max_steps = 20 #20
        self.history_window = 10 #10
        self.history = []
        self.workers = {
            "browser_use_agent": BrowserUseAgent(True),
            #"arxiv_search": AutoTool.from_preloaded("example/arxiv")
        }
    
    def run(self, task_input: str):
        
        llms = [    
            {
                #"name": "gpt-4o", #kws
                "name": "gpt-4o-mini",
                "backend": "openai"
            }
        ]
        
        WORKER_PROMPTS = """
1. `self.workers["browser_use_agent"].run(task_input: str, start_url: str)`: Call the browser use agent to solve the given task and start from the given url.
        """
        
        system_prompt = f"""# Task Orchestration Instructions

You are an orchestrator agent responsible for coordinating specialized workers to solve complex tasks. Your goal is to break down the main task into subtasks and assign them to appropriate workers.

## Main Task
{task_input}

## Available Workers
{WORKER_PROMPTS}

## Your Responsibilities:
1. **Analyze the task** and break it down into logical subtasks
2. **Assign subtasks** to appropriate workers from your available list
3. **Coordinate the workflow** by processing each worker's output
4. **Synthesize results** into a comprehensive solution
5. **Verify completeness** before finalizing

## Solution Requirements:
- Provide detailed explanations for each step
- Include specific implementations and examples where appropriate
- Ensure your solution directly addresses the original task
- If one approach fails, try alternative methods

## Completion Protocol:
- Before submitting your final answer, double-check that you've fully completed the task
- Verify your solution against the original requirements
- When you're confident the task is complete, format your answer as:
  ```
  <FINAL_ANSWER>[brief one-line summary of the task result]</FINAL_ANSWER>
  ```
- Only use the FINAL_ANSWER tag when the task is truly complete

## Problem-Solving Tips:
- Try multiple approaches if your first method fails
- For web searches, check Wikipedia first before exploring other sources
- When searching, use advanced filters when appropriate (date, location, etc.)
- For math problems, consider using Python with the sympy library
- Always verify your answers through cross-checking
- Don't rely solely on your knowledge - use available tools
- When executing code, debug any errors rather than assuming correct results
- Search results rarely provide complete answers - use them to find sources for further analysis
- For file downloads, use web browser simulation or write appropriate code"""
        
        messages = [
            {"content": system_prompt, "role": "system"}
        ]
        
        final_answer = ""
        
        # breakpoint()
        rounds = 0
        
        while rounds < self.max_steps:
            step_instructions = f"""
## Step-by-Step Execution Protocol:
Here are the latest {self.history_window} trajectory (at most) you have taken:
<history>
{self.history[-self.history_window:]}
</history>

Your output should be in json format, including the following fields:
- `observation`: Do not over-confident about the correctness of the history actions. You should
always check the current state to make sure the correctness of the next action.
- `reasoning`: The reasoning about the next action you want to take, and the possible obstacles you may encounter, 
and how to solve them. Do not forget to check the history actions to avoid the same mistakes.
- `worker_name`: The name of the worker you want to use. It is only one step action
without any other texts (such as annotation)
- `worker_params`: The parameters for the worker. It is a dictionary containing the parameters for the worker.

When you have recognized the task is completed, you need to output 

Here is two example of the output:
```json
{{
    "observation": "I have obtained the answer FeO is the densest iron oxide on the Moon...",
    "reasoning": "Since I have already obtained the answer, I do not need to call any worker... ",
    "worker_name": None,
    "worker_params": None
}}

{{
    "observation":  "At current stage, I have already opened the amazon website...",
    "reasoning": "To proceed with the task of searching for iphone products, I need to complete...",
    "worker_name": "browser_use_agent",
    "worker_params": {{
        "task_prompt": "search for the product 'iphone' on amazon",
        "start_url": "https://www.amazon.com"
    }}
}}           """
            messages.append({"content": step_instructions, "role": "user"})
            
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "orchestration",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "observation": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "worker_name": {"type": "string"},
                            "worker_params": {"type": "object"}
                        },
                        "required": ["observation", "reasoning", "worker_name", "worker_params"]
                    }
                }
            }
            
            response = llm_chat_with_json_output(
                agent_name=self.agent_name,
                messages=messages,
                llms=llms,
                response_format=response_format
            )
                        
            step_response = response["response"]["response_message"]
            
            resp_dict = _parse_json_output(step_response)
            observation = resp_dict.get("observation", "")
            reasoning = resp_dict.get("reasoning", "")
            worker_name = resp_dict.get("worker_name", None)
            worker_params = resp_dict.get("worker_params", {})
            
            if worker_name is None:
                if worker_params is {}:
                    trajectory_info = {
                        "round": rounds,
                        "observation": observation,
                        "thought": reasoning,
                        "called_worker": worker_name,
                        "called_worker_params": worker_params,
                        "info": None
                    }
                    final_answer = self.get_final_answer(task_input)
                
                break
                
            else:
                
                worker_response = self.workers[worker_name].run(**worker_params)
                
                result = worker_response["result"]
                
                trajectory_info = {
                    "round": rounds,
                    "observation": observation,
                    "thought": reasoning,
                    "called_worker": worker_name,
                    "called_worker_params": worker_params,
                    "info": result
                }
                
                print(trajectory_info)
            
                self.history.append(trajectory_info)
                
            rounds += 1
        
        return {
            "agent_name": self.agent_name,
            "result": final_answer,
            "rounds": rounds
        }
    
    def get_final_answer(self, task_input: str):
        r"""Get the final answer based on the task prompt and current state.
        It is used when the agent thinks that the task can be completed without any further action, and answer can be directly found in the current state.
        """
        system_prompt = """
You are an extractor agent. Your job is to extract the final answer from the history and the task prompt.
"""
        prompt = f"""
You are solving a complex task which needs multi-step interaction with different workers. After the multi-step observation, reasoning and acting taken by different workers, you thinkthe task is currently solved.
Here are all trajectory we have taken:
<history>{self.history}</history>
Please find the final answer, or give valuable insights and founds (e.g. if previous actions contain downloading files, your output should include the path of the downloaded file) about the overall task: <task>{task_input}</task>
        """

        messages = [
            {"content": system_prompt, "role": "system"},
            {"content": prompt, "role": "user"}
        ]
        
        llms = [
            {
                "name": "gpt-4o-mini",
                "backend": "openai"
            }
        ]

        response = llm_chat(
            agent_name=self.agent_name,
            messages=messages,
            llms=llms
        )
        
        return response["response"]["response_message"]

def main():
    agent = ReActAgent()
    
    data = {
        "Question": """
What is the temperature difference between Edison and New York today? 
""",
        "Tools": "1. Web browser, 2. Calculator"
    }
    
    main_parser = get_parser()
    main_args = main_parser.parse_args()
    dataset = load_dataset(main_args.data_name, "2023_all", split=main_args.split)
    
    dataset = dataset["Question"]
    
    # for idx, question in enumerate(dataset):
    #     result = agent.run(question)
    #     print(result)
    result = agent.run(data["Question"])
    print(result)
    
if __name__ == "__main__":
    
    main()


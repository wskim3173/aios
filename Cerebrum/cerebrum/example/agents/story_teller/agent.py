from cerebrum.llm.apis import llm_chat, llm_call_tool, llm_chat_with_json_output
from cerebrum.interface import AutoTool
import json
import os
from cerebrum.config.config_manager import config

aios_kernel_url = config.get_kernel_url()

class StoryTeller:
    def __init__(self, agent_name):
        self.agent_name = agent_name
        self.config = self.load_config()
        self.tools = [
            tool.get_tool_call_format()
            for tool in AutoTool.from_batch_preloaded(self.config["tools"])
        ]

        self.plan_max_fail_times = 3
        self.tool_call_max_fail_times = 3

        self.start_time = None
        self.end_time = None
        self.request_waiting_times: list = []
        self.request_turnaround_times: list = []
        self.messages = []
        self.workflow_mode = "manual"  # (manual, automatic)
        self.rounds = 0

    def load_config(self):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        config_file = os.path.join(script_dir, "config.json")

        with open(config_file, "r") as f:
            config = json.load(f)
            
        return config
    
    def pre_select_tools(self, tool_names):
        pre_selected_tools = []
        for tool_name in tool_names:
            for tool in self.tools:
                if tool["function"]["name"] == tool_name:
                    pre_selected_tools.append(tool)
                    break  
        return pre_selected_tools
    def build_system_instruction(self):
        prefix = "".join(["".join(self.config["description"])])

        plan_instruction = "".join(
            [
                f"You are given the available tools from the tool list: {json.dumps(self.tools)} to help you solve problems. ",
                "Generate a plan with comprehensive yet minimal steps to fulfill the task. ",
                "The plan must follow the json format as below: ",
                "[",
                '{"action_type": "action_type_value", "action": "action_value","tool_use": [tool_name1, tool_name2,...]}',
                '{"action_type": "action_type_value", "action": "action_value", "tool_use": [tool_name1, tool_name2,...]}',
                "...",
                "]",
                "In each step of the planned plan, identify tools to use and recognize no tool is necessary. ",
                "Followings are some plan examples. ",
                "[" "[",
                '{"action_type": "tool_use", "action": "gather information from arxiv. ", "tool_use": ["arxiv"]},',
                '{"action_type": "chat", "action": "write a summarization based on the gathered information. ", "tool_use": []}',
                "];",
                "[",
                '{"action_type": "tool_use", "action": "gather information from arxiv. ", "tool_use": ["arxiv"]},',
                '{"action_type": "chat", "action": "understand the current methods and propose ideas that can improve ", "tool_use": []}',
                "]",
                "]",
            ]
        )

        if self.workflow_mode == "manual":
            self.messages.append({"role": "system", "content": prefix})

        else:
            assert self.workflow_mode == "automatic"
            self.messages.append({"role": "system", "content": prefix})
            self.messages.append({"role": "user", "content": plan_instruction})

    def automatic_workflow(self):
        for i in range(self.plan_max_fail_times):
            response = llm_chat_with_json_output(
                messages=self.messages,
                message_return_type="json"
            )["response"]["response_message"]

            try:
                workflow = json.loads(response)
            except:
                workflow = None

            self.rounds += 1

            if workflow:
                return workflow

            else:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": f"Fail {i+1} times to generate a valid plan. I need to regenerate a plan",
                    }
                )
        return None

    def manual_workflow(self):
        workflow = [
            {
                "action_type": "chat",
                "action": "Determine the story genre and theme based on user input and generate initial story plot and characters.",
                "tool_use": []
            },
            {
                "action_type": "call_tool",
                "action": "Create visual representations for the main character.",
                "tool_use": []
            },
            {
                "action_type": "chat",
                "action": "Incorporate it into the story narrative and summarize the story",
                "tool_use": []
            }
        ]
        return workflow

    def run(self, task_input):
        self.build_system_instruction()


        self.messages.append({"role": "user", "content": task_input})

        workflow = None

        if self.workflow_mode == "automatic":
            workflow = self.automatic_workflow()
            self.messages = self.messages[:1]  # clear long context

        else:
            assert self.workflow_mode == "manual"
            workflow = self.manual_workflow()

        self.messages.append(
            {
                "role": "user",
                "content": f"[Thinking]: The workflow generated for the problem is {json.dumps(workflow)}. Follow the workflow to solve the problem step by step. ",
            }
        )

        try:
            if workflow:
                final_result = ""

                for i, step in enumerate(workflow):
                    action_type = step["action_type"]
                    action = step["action"]
                    tool_use = step["tool_use"]

                    prompt = f"At step {i + 1}, you need to: {action}. "
                    self.messages.append({"role": "user", "content": prompt})

                    if tool_use:
                        selected_tools = self.pre_select_tools(tool_use)

                    else:
                        selected_tools = None

                    if action_type == "call_tool":
                        response = llm_call_tool(
                            agent_name=self.agent_name,
                            messages=self.messages,
                            tools=selected_tools,
                            base_url=aios_kernel_url
                        )["response"]
                    else:
                        response = llm_chat(
                            agent_name=self.agent_name,
                            messages=self.messages,
                            base_url=aios_kernel_url
                        )["response"]
                    
                    self.messages.append({"role": "assistant", "content": response["response_message"]})

                    self.rounds += 1


                final_result = self.messages[-1]["content"]
                
                return {
                    "agent_name": self.agent_name,
                    "result": final_result,
                    "rounds": self.rounds,
                }

            else:
                return {
                    "agent_name": self.agent_name,
                    "result": "Failed to generate a valid workflow in the given times.",
                    "rounds": self.rounds,

                }
                
        except Exception as e:

            return {}

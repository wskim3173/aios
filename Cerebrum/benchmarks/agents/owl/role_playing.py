from typing import Dict, List, Optional, Tuple


from camel.agents import ChatAgent
from camel.responses import ChatAgentResponse
from camel.messages.base import BaseMessage
from camel.societies import RolePlaying
from camel.logger import get_logger


from copy import deepcopy

logger = get_logger(__name__)


class OwlRolePlaying(RolePlaying):
    def __init__(self, **kwargs):
        self.user_role_name = kwargs.get("user_role_name", "user")
        self.assistant_role_name = kwargs.get("assistant_role_name", "assistant")

        self.output_language = kwargs.get("output_language", None)

        self.user_agent_kwargs: dict = kwargs.get("user_agent_kwargs", {})
        self.assistant_agent_kwargs: dict = kwargs.get("assistant_agent_kwargs", {})

        self.output_language = kwargs.get("output_language", None)

        super().__init__(**kwargs)

        init_user_sys_msg, init_assistant_sys_msg = self._construct_gaia_sys_msgs()

        self.assistant_agent: ChatAgent
        self.user_agent: ChatAgent
        self.assistant_sys_msg: Optional[BaseMessage]
        self.user_sys_msg: Optional[BaseMessage]

        # self.is_reasoning_task = self._judge_if_reasoning_task(self.task_prompt)

        # if self.is_reasoning_task:
        #     logger.info("The task is judged as a reasoning or coding task. The assistant agent will use the reasoning model O3-MINI.")
        # else:
        #     logger.info("The assistant agent will use the default model.")

        self._init_agents(
            init_assistant_sys_msg,
            init_user_sys_msg,
            assistant_agent_kwargs=self.assistant_agent_kwargs,
            user_agent_kwargs=self.user_agent_kwargs,
            output_language=self.output_language,
            # is_reasoning_task=self.is_reasoning_task
        )

    def _init_agents(
        self,
        init_assistant_sys_msg: BaseMessage,
        init_user_sys_msg: BaseMessage,
        assistant_agent_kwargs: Optional[Dict] = None,
        user_agent_kwargs: Optional[Dict] = None,
        output_language: Optional[str] = None,
        is_reasoning_task: bool = False,
    ) -> None:
        r"""Initialize assistant and user agents with their system messages.

        Args:
            init_assistant_sys_msg (BaseMessage): Assistant agent's initial
                system message.
            init_user_sys_msg (BaseMessage): User agent's initial system
                message.
            assistant_agent_kwargs (Dict, optional): Additional arguments to
                pass to the assistant agent. (default: :obj:`None`)
            user_agent_kwargs (Dict, optional): Additional arguments to
                pass to the user agent. (default: :obj:`None`)
            output_language (str, optional): The language to be output by the
                agents. (default: :obj:`None`)
        """
        if self.model is not None:
            if assistant_agent_kwargs is None:
                assistant_agent_kwargs = {"model": self.model}
            elif "model" not in assistant_agent_kwargs:
                assistant_agent_kwargs.update(dict(model=self.model))
            if user_agent_kwargs is None:
                user_agent_kwargs = {"model": self.model}
            elif "model" not in user_agent_kwargs:
                user_agent_kwargs.update(dict(model=self.model))

        # # If the task is a reasoning task, the assistant agent should use the reasoning model O3-MINI
        # if is_reasoning_task:
        #     assistant_agent_kwargs['model'] = ModelFactory.create(
        #         model_platform=ModelPlatformType.OPENAI,
        #         model_type=ModelType.O3_MINI,
        #     )

        self.assistant_agent = ChatAgent(
            init_assistant_sys_msg,
            output_language=output_language,
            **(assistant_agent_kwargs or {}),
        )
        self.assistant_sys_msg = self.assistant_agent.system_message

        self.user_agent = ChatAgent(
            init_user_sys_msg,
            output_language=output_language,
            **(user_agent_kwargs or {}),
        )
        self.user_sys_msg = self.user_agent.system_message

    # def _judge_if_reasoning_task(self, question: str) -> bool:
    #     r"""Judge if the question is a reasoning task."""

    #     LLM = OpenAIModel(model_type=ModelType.O3_MINI)
    #     prompt = f"""
    #     Please judge whether the following question is a reasoning or coding task, which can be solved by reasoning without leveraging external resources, or is suitable for writing code to solve the task.
    #     If it is a reasoning or coding task, please return only "yes".
    #     If it is not a reasoning or coding task, please return only "no".
    #     Note:
    #     - If the question required some world knowledge to answer the question, please carefully judge it, because the model's own knowledge is often unreliable.
    #     - If it is suitable for writing codes (e.g. process excel files, write simulation codes, etc.), in most cases, it can be considered as a coding task.
    #     Question: <question>{question}</question>
    #     """
    #     messages = [{"role": "user", "content": prompt}]
    #     resp = LLM.run(messages)
    #     if 'yes' in resp.choices[0].message.content.lower():
    #         return True
    #     else:
    #         return False

    def _construct_gaia_sys_msgs(self):
        user_system_prompt = f"""
===== RULES OF USER =====
Never forget you are a user and I am a assistant. Never flip roles! You will always instruct me. We share a common interest in collaborating to successfully complete a task.
I must help you to complete a difficult task.
You must instruct me based on my expertise and your needs to solve the task step by step. The format of your instruction is: `Instruction: [YOUR INSTRUCTION]`, where "Instruction" describes a sub-task or question.
You must give me one instruction at a time.
I must write a response that appropriately solves the requested instruction.
You should instruct me not ask me questions.

Please note that the task may be very complicated. Do not attempt to solve the task by single step. You must instruct me to find the answer step by step.
Here are some tips that will help you to give more valuable instructions about our task to me:
<tips>
- I have various tools to use, such as search toolkit, web browser simulation toolkit, document relevant toolkit, code execution toolkit, etc. Thus, You must think how human will solve the task step-by-step, and give me instructions just like that. For example, one may first use google search to get some initial information and the target url, then retrieve the content of the url, or do some web browser interaction to find the answer.
- Although the task is complex, the answer does exist. If you can't find the answer using the current scheme, try to re-plan and use other ways to find the answer, e.g. using other tools or methods that can achieve similar results.
- Always remind me to verify my final answer about the overall task. This work can be done by using multiple tools(e.g., screenshots, webpage analysis, etc.), or something else.
- If I have written code, please remind me to run the code and get the result.
- Search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding sources rather than direct answers, as it always need to use other tools to further process the url, e.g. interact with the webpage, extract webpage content, etc. 
- If the question mentions youtube video, in most cases you have to process the content of the mentioned video.
- For downloading files, you can either use the web browser simulation toolkit or write codes (for example, the github content can be downloaded via https://raw.githubusercontent.com/...).
- Flexibly write codes to solve some problems, such as excel relevant tasks.
</tips>

Now, here is the overall task: <task>{self.task_prompt}</task>. Never forget our task!

Now you must start to instruct me to solve the task step-by-step. Do not add anything else other than your instruction!
Keep giving me instructions until you think the task is completed.
When the task is completed, you must only reply with a single word <TASK_DONE>.
Never say <TASK_DONE> unless my responses have solved your task.
        """

        assistant_system_prompt = f"""
===== RULES OF ASSISTANT =====
Never forget you are a assistant and I am a user. Never flip roles! Never instruct me! You have to utilize your available tools to solve the task I assigned.
We share a common interest in collaborating to successfully complete a complex task.
You must help me to complete the task.

Here is our overall task: {self.task_prompt}. Never forget our task!

I must instruct you based on your expertise and my needs to complete the task. An instruction is typically a sub-task or question.

You must leverage your available tools, try your best to solve the problem, and explain your solutions.
Unless I say the task is completed, you should always start with:
Solution: [YOUR_SOLUTION]
[YOUR_SOLUTION] should be specific, including detailed explanations and provide preferable detailed implementations and examples and lists for task-solving.

Please note that our overall task may be very complicated. Here are some tips that may help you solve the task:
<tips>
- If one way fails to provide an answer, try other ways or methods. The answer does exists.
- If the search snippet is unhelpful but the URL comes from an authoritative source, try visit the website for more details.  
- When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.  
- When solving tasks that require web searches, check Wikipedia first before exploring other websites.  
- When trying to solve math problems, you can try to write python code and use sympy library to solve the problem.
- Always verify the accuracy of your final answers! Try cross-checking the answers by other ways. (e.g., screenshots, webpage analysis, etc.).  
- Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.  
- After writing codes, do not forget to run the code and get the result. If it encounters an error, try to debug it. Also, bear in mind that the code execution environment does not support interactive input.
- When a tool fails to run, or the code does not run correctly, never assume that it returns the correct result and continue to reason based on the assumption, because the assumed result cannot lead you to the correct answer. The right way is to think about the reason for the error and try again.
- Search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding sources rather than direct answers, as it always need to use other tools to further process the url, e.g. interact with the webpage, extract webpage content, etc. 
- For downloading files, you can either use the web browser simulation toolkit or write codes.
</tips>

        """

        user_sys_msg = BaseMessage.make_user_message(
            role_name=self.user_role_name, content=user_system_prompt
        )

        assistant_sys_msg = BaseMessage.make_assistant_message(
            role_name=self.assistant_role_name, content=assistant_system_prompt
        )

        return user_sys_msg, assistant_sys_msg

    def step(
        self, assistant_msg: BaseMessage
    ) -> Tuple[ChatAgentResponse, ChatAgentResponse]:
        user_response = self.user_agent.step(assistant_msg)
        if user_response.terminated or user_response.msgs is None:
            return (
                ChatAgentResponse(msgs=[], terminated=False, info={}),
                ChatAgentResponse(
                    msgs=[],
                    terminated=user_response.terminated,
                    info=user_response.info,
                ),
            )
        user_msg = self._reduce_message_options(user_response.msgs)

        modified_user_msg = deepcopy(user_msg)

        if "TASK_DONE" not in user_msg.content:
            modified_user_msg.content += f"""\n
            Here are auxiliary information about the overall task, which may help you understand the intent of the current task:
            <auxiliary_information>
            {self.task_prompt}
            </auxiliary_information>
            If there are available tools and you want to call them, never say 'I will ...', but first call the tool and reply based on tool call's result, and tell me which tool you have called.
            """

        else:
            # The task is done, and the assistant agent need to give the final answer about the original task
            modified_user_msg.content += f"""\n
            Now please make a final answer of the original task based on our conversation : <task>{self.task_prompt}</task>
            """

        # process assistant's response
        assistant_response = self.assistant_agent.step(modified_user_msg)
        if assistant_response.terminated or assistant_response.msgs is None:
            return (
                ChatAgentResponse(
                    msgs=[],
                    terminated=assistant_response.terminated,
                    info=assistant_response.info,
                ),
                ChatAgentResponse(
                    msgs=[user_msg], terminated=False, info=user_response.info
                ),
            )
        assistant_msg = self._reduce_message_options(assistant_response.msgs)

        modified_assistant_msg = deepcopy(assistant_msg)
        if "TASK_DONE" not in user_msg.content:
            modified_assistant_msg.content += f"""\n
                Provide me with the next instruction and input (if needed) based on my response and our current task: <task>{self.task_prompt}</task>
                Before producing the final answer, please check whether I have rechecked the final answer using different toolkit as much as possible. If not, please remind me to do that.
                If I have written codes, remind me to run the codes.
                If you think our task is done, reply with `TASK_DONE` to end our conversation.
            """

        # return the modified messages
        return (
            ChatAgentResponse(
                msgs=[modified_assistant_msg],
                terminated=assistant_response.terminated,
                info=assistant_response.info,
            ),
            ChatAgentResponse(
                msgs=[modified_user_msg],
                terminated=user_response.terminated,
                info=user_response.info,
            ),
        )


class OwlGAIARolePlaying(OwlRolePlaying):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(
        self, assistant_msg: BaseMessage
    ) -> Tuple[ChatAgentResponse, ChatAgentResponse]:
        user_response = self.user_agent.step(assistant_msg)
        if user_response.terminated or user_response.msgs is None:
            return (
                ChatAgentResponse(msgs=[], terminated=False, info={}),
                ChatAgentResponse(
                    msgs=[],
                    terminated=user_response.terminated,
                    info=user_response.info,
                ),
            )
        user_msg = self._reduce_message_options(user_response.msgs)

        modified_user_msg = deepcopy(user_msg)

        if "TASK_DONE" not in user_msg.content:
            modified_user_msg.content += f"""\n
            Here are auxiliary information about the overall task, which may help you understand the intent of the current task:
            <auxiliary_information>
            {self.task_prompt}
            </auxiliary_information>
            If there are available tools and you want to call them, never say 'I will ...', but first call the tool and reply based on tool call's result, and tell me which tool you have called.
            """

        else:
            # The task is done, and the assistant agent need to give the final answer about the original task
            modified_user_msg.content += f"""\n
            Now please make a final answer of the original task based on our conversation : <task>{self.task_prompt}</task>
            Please pay special attention to the format in which the answer is presented.
            You should first analyze the answer format required by the question and then output the final answer that meets the format requirements. 
            Your response should include the following content:
            - `analysis`: enclosed by <analysis> </analysis>, a detailed analysis of the reasoning result.
            - `final_answer`: enclosed by <final_answer> </final_answer>, the final answer to the question.
            Here are some hint about the final answer:
            <hint>
            Your final answer must be output exactly in the format specified by the question. It should be a number OR as few words as possible OR a comma separated list of numbers and/or strings:
            - If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. 
            - If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. 
            - If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
            </hint>
            """

        # process assistant's response
        assistant_response = self.assistant_agent.step(modified_user_msg)
        if assistant_response.terminated or assistant_response.msgs is None:
            return (
                ChatAgentResponse(
                    msgs=[],
                    terminated=assistant_response.terminated,
                    info=assistant_response.info,
                ),
                ChatAgentResponse(
                    msgs=[user_msg], terminated=False, info=user_response.info
                ),
            )
        assistant_msg = self._reduce_message_options(assistant_response.msgs)

        modified_assistant_msg = deepcopy(assistant_msg)
        if "TASK_DONE" not in user_msg.content:
            modified_assistant_msg.content += f"""\n
                Provide me with the next instruction and input (if needed) based on my response and our current task: <task>{self.task_prompt}</task>
                Before producing the final answer, please check whether I have rechecked the final answer using different toolkit as much as possible. If not, please remind me to do that.
                If I have written codes, remind me to run the codes.
                If you think our task is done, reply with `TASK_DONE` to end our conversation.
            """

        # return the modified messages
        return (
            ChatAgentResponse(
                msgs=[modified_assistant_msg],
                terminated=assistant_response.terminated,
                info=assistant_response.info,
            ),
            ChatAgentResponse(
                msgs=[modified_user_msg],
                terminated=user_response.terminated,
                info=user_response.info,
            ),
        )


def run_society(
    society: OwlRolePlaying,
    round_limit: int = 15,
) -> Tuple[str, List[dict], dict]:
    overall_completion_token_count = 0
    overall_prompt_token_count = 0

    chat_history = []
    init_prompt = """
    Now please give me instructions to solve over overall task step by step. If the task requires some specific knowledge, please instruct me to use tools to complete the task.
        """
    input_msg = society.init_chat(init_prompt)
    for _round in range(round_limit):
        assistant_response, user_response = society.step(input_msg)
        # Check if usage info is available before accessing it
        if assistant_response.info.get("usage") and user_response.info.get("usage"):
            overall_completion_token_count += assistant_response.info["usage"].get(
                "completion_tokens", 0
            ) + user_response.info["usage"].get("completion_tokens", 0)
            overall_prompt_token_count += assistant_response.info["usage"].get(
                "prompt_tokens", 0
            ) + user_response.info["usage"].get("prompt_tokens", 0)

        # convert tool call to dict
        tool_call_records: List[dict] = []
        if assistant_response.info.get("tool_calls"):
            for tool_call in assistant_response.info["tool_calls"]:
                tool_call_records.append(tool_call.as_dict())

        _data = {
            "user": user_response.msg.content
            if hasattr(user_response, "msg") and user_response.msg
            else "",
            "assistant": assistant_response.msg.content
            if hasattr(assistant_response, "msg") and assistant_response.msg
            else "",
            "tool_calls": tool_call_records,
        }

        chat_history.append(_data)
        logger.info(
            f"Round #{_round} user_response:\n {user_response.msgs[0].content if user_response.msgs and len(user_response.msgs) > 0 else ''}"
        )
        logger.info(
            f"Round #{_round} assistant_response:\n {assistant_response.msgs[0].content if assistant_response.msgs and len(assistant_response.msgs) > 0 else ''}"
        )

        if (
            assistant_response.terminated
            or user_response.terminated
            or "TASK_DONE" in user_response.msg.content
        ):
            break

        input_msg = assistant_response.msg

    answer = chat_history[-1]["assistant"]
    token_info = {
        "completion_token_count": overall_completion_token_count,
        "prompt_token_count": overall_prompt_token_count,
    }

    return answer, chat_history, token_info


import logging
from typing import Dict, List, Optional, Sequence, Tuple, Union

from camel.agents import (
    ChatAgent,
    CriticAgent,
    TaskPlannerAgent,
    TaskSpecifyAgent,
)
from camel.generators import SystemMessageGenerator
from camel.human import Human
from camel.messages import BaseMessage
from camel.models import BaseModelBackend
from camel.prompts import TextPrompt
from camel.responses import ChatAgentResponse
from camel.types import RoleType, TaskType

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class RolePlaying:
    r"""Role playing between two agents.

    Args:
        assistant_role_name (str): The name of the role played by the
            assistant.
        user_role_name (str): The name of the role played by the user.
        critic_role_name (str, optional): The name of the role played by the
            critic. Role name with :obj:`"human"` will set critic as a
            :obj:`Human` agent, else will create a :obj:`CriticAgent`.
            (default: :obj:`"critic"`)
        task_prompt (str, optional): A prompt for the task to be performed.
            (default: :obj:`""`)
        with_task_specify (bool, optional): Whether to use a task specify
            agent. (default: :obj:`True`)
        with_task_planner (bool, optional): Whether to use a task planner
            agent. (default: :obj:`False`)
        with_critic_in_the_loop (bool, optional): Whether to include a critic
            in the loop. (default: :obj:`False`)
        critic_criteria (str, optional): Critic criteria for the critic agent.
            If not specified, set the criteria to improve task performance.
        model (BaseModelBackend, optional): The model backend to use for
            generating responses. If specified, it will override the model in
            all agents if not specified in agent-specific kwargs. (default:
            :obj:`OpenAIModel` with `GPT_4O_MINI`)
        task_type (TaskType, optional): The type of task to perform.
            (default: :obj:`TaskType.AI_SOCIETY`)
        assistant_agent_kwargs (Dict, optional): Additional arguments to pass
            to the assistant agent. (default: :obj:`None`)
        user_agent_kwargs (Dict, optional): Additional arguments to pass to
            the user agent. (default: :obj:`None`)
        task_specify_agent_kwargs (Dict, optional): Additional arguments to
            pass to the task specify agent. (default: :obj:`None`)
        task_planner_agent_kwargs (Dict, optional): Additional arguments to
            pass to the task planner agent. (default: :obj:`None`)
        critic_kwargs (Dict, optional): Additional arguments to pass to the
            critic. (default: :obj:`None`)
        sys_msg_generator_kwargs (Dict, optional): Additional arguments to
            pass to the system message generator. (default: :obj:`None`)
        extend_sys_msg_meta_dicts (List[Dict], optional): A list of dicts to
            extend the system message meta dicts with. (default: :obj:`None`)
        extend_task_specify_meta_dict (Dict, optional): A dict to extend the
            task specify meta dict with. (default: :obj:`None`)
        output_language (str, optional): The language to be output by the
            agents. (default: :obj:`None`)
    """

    def __init__(
        self,
        assistant_role_name: str,
        user_role_name: str,
        *,
        critic_role_name: str = "critic",
        task_prompt: str = "",
        with_task_specify: bool = True,
        with_task_planner: bool = False,
        with_critic_in_the_loop: bool = False,
        critic_criteria: Optional[str] = None,
        model: Optional[BaseModelBackend] = None,
        task_type: TaskType = TaskType.AI_SOCIETY,
        assistant_agent_kwargs: Optional[Dict] = None,
        user_agent_kwargs: Optional[Dict] = None,
        task_specify_agent_kwargs: Optional[Dict] = None,
        task_planner_agent_kwargs: Optional[Dict] = None,
        critic_kwargs: Optional[Dict] = None,
        sys_msg_generator_kwargs: Optional[Dict] = None,
        extend_sys_msg_meta_dicts: Optional[List[Dict]] = None,
        extend_task_specify_meta_dict: Optional[Dict] = None,
        output_language: Optional[str] = None,
    ) -> None:
        if model is not None:
            logger.warning(
                "Model provided globally is set for all agents if not"
                " already specified in agent_kwargs."
            )

        self.with_task_specify = with_task_specify
        self.with_task_planner = with_task_planner
        self.with_critic_in_the_loop = with_critic_in_the_loop
        self.model = model
        self.task_type = task_type
        self.task_prompt = task_prompt

        self.specified_task_prompt: Optional[TextPrompt] = None
        self._init_specified_task_prompt(
            assistant_role_name,
            user_role_name,
            task_specify_agent_kwargs=task_specify_agent_kwargs,
            extend_task_specify_meta_dict=extend_task_specify_meta_dict,
            output_language=output_language,
        )

        self.planned_task_prompt: Optional[TextPrompt] = None
        self._init_planned_task_prompt(
            task_planner_agent_kwargs=task_planner_agent_kwargs,
            output_language=output_language,
        )

        sys_msg_generator = SystemMessageGenerator(
            task_type=self.task_type,
            **(sys_msg_generator_kwargs or {}),
        )

        (
            init_assistant_sys_msg,
            init_user_sys_msg,
            sys_msg_meta_dicts,
        ) = self._get_sys_message_info(
            assistant_role_name,
            user_role_name,
            sys_msg_generator,
            extend_sys_msg_meta_dicts=extend_sys_msg_meta_dicts,
        )

        self.assistant_agent: ChatAgent
        self.user_agent: ChatAgent
        self.assistant_sys_msg: Optional[BaseMessage]
        self.user_sys_msg: Optional[BaseMessage]
        self._init_agents(
            init_assistant_sys_msg,
            init_user_sys_msg,
            assistant_agent_kwargs=assistant_agent_kwargs,
            user_agent_kwargs=user_agent_kwargs,
            output_language=output_language,
        )
        self.critic: Optional[Union[CriticAgent, Human]] = None
        self.critic_sys_msg: Optional[BaseMessage] = None
        self._init_critic(
            sys_msg_generator,
            sys_msg_meta_dicts,
            critic_role_name,
            critic_criteria=critic_criteria,
            critic_kwargs=critic_kwargs,
        )

    def _init_specified_task_prompt(
        self,
        assistant_role_name: str,
        user_role_name: str,
        task_specify_agent_kwargs: Optional[Dict] = None,
        extend_task_specify_meta_dict: Optional[Dict] = None,
        output_language: Optional[str] = None,
    ) -> None:
        r"""Use a task specify agent to generate a specified task prompt.
        Generated specified task prompt will be used to replace original
        task prompt. If there is no task specify agent, specified task
        prompt will not be generated.

        Args:
            assistant_role_name (str): The name of the role played by the
                assistant.
            user_role_name (str): The name of the role played by the user.
            task_specify_agent_kwargs (Dict, optional): Additional arguments
                to pass to the task specify agent. (default: :obj:`None`)
            extend_task_specify_meta_dict (Dict, optional): A dict to extend
                the task specify meta dict with. (default: :obj:`None`)
            output_language (str, optional): The language to be output by the
                agents. (default: :obj:`None`)
        """
        if self.with_task_specify:
            task_specify_meta_dict = dict()
            if self.task_type in [TaskType.AI_SOCIETY, TaskType.MISALIGNMENT]:
                task_specify_meta_dict.update(
                    dict(
                        assistant_role=assistant_role_name,
                        user_role=user_role_name,
                    )
                )
            task_specify_meta_dict.update(extend_task_specify_meta_dict or {})
            if self.model is not None:
                if task_specify_agent_kwargs is None:
                    task_specify_agent_kwargs = {'model': self.model}
                elif 'model' not in task_specify_agent_kwargs:
                    task_specify_agent_kwargs.update(dict(model=self.model))
            task_specify_agent = TaskSpecifyAgent(
                task_type=self.task_type,
                output_language=output_language,
                **(task_specify_agent_kwargs or {}),
            )
            self.specified_task_prompt = task_specify_agent.run(
                self.task_prompt,
                meta_dict=task_specify_meta_dict,
            )
            self.task_prompt = self.specified_task_prompt

    def _init_planned_task_prompt(
        self,
        task_planner_agent_kwargs: Optional[Dict] = None,
        output_language: Optional[str] = None,
    ) -> None:
        r"""Use a task plan agent to append a planned task prompt to task
        prompt. The planned task prompt is generated based on the task
        prompt, which can be original task prompt or specified task prompt
        if available. If there is no task plan agent, planned task prompt
        will not be generated.

        Args:
            task_planner_agent_kwargs (Dict, optional): Additional arguments
                to pass to the task planner agent. (default: :obj:`None`)
            output_language (str, optional): The language to be output by the
                agents. (default: :obj:`None`)
        """
        if self.with_task_planner:
            if self.model is not None:
                if task_planner_agent_kwargs is None:
                    task_planner_agent_kwargs = {'model': self.model}
                elif 'model' not in task_planner_agent_kwargs:
                    task_planner_agent_kwargs.update(dict(model=self.model))
            task_planner_agent = TaskPlannerAgent(
                output_language=output_language,
                **(task_planner_agent_kwargs or {}),
            )
            self.planned_task_prompt = task_planner_agent.run(self.task_prompt)
            self.task_prompt = (
                f"{self.task_prompt}\n" f"{self.planned_task_prompt}"
            )
        else:
            self.planned_task_prompt = None

    def _get_sys_message_info(
        self,
        assistant_role_name: str,
        user_role_name: str,
        sys_msg_generator: SystemMessageGenerator,
        extend_sys_msg_meta_dicts: Optional[List[Dict]] = None,
    ) -> Tuple[BaseMessage, BaseMessage, List[Dict]]:
        r"""Get initial assistant and user system message with a list of
        system message meta dicts.

        Args:
            assistant_role_name (str): The name of the role played by the
                assistant.
            user_role_name (str): The name of the role played by the user.
            sys_msg_generator (SystemMessageGenerator): A system message
                generator for agents.
            extend_sys_msg_meta_dicts (List[Dict], optional): A list of dicts
                to extend the system message meta dicts with.
                (default: :obj:`None`)

        Returns:
            Tuple[BaseMessage, BaseMessage, List[Dict]]: A tuple containing a
                `BaseMessage` representing the assistant's initial system
                message, a `BaseMessage` representing the user's initial system
                message, and a list of system message meta dicts.
        """
        sys_msg_meta_dicts = [dict(task=self.task_prompt) for _ in range(2)]
        if extend_sys_msg_meta_dicts is None and self.task_type in [
            TaskType.AI_SOCIETY,
            TaskType.MISALIGNMENT,
        ]:
            extend_sys_msg_meta_dicts = [
                dict(
                    assistant_role=assistant_role_name,
                    user_role=user_role_name,
                )
                for _ in range(2)
            ]

        if extend_sys_msg_meta_dicts is not None:
            sys_msg_meta_dicts = [
                {**sys_msg_meta_dict, **extend_sys_msg_meta_dict}
                for sys_msg_meta_dict, extend_sys_msg_meta_dict in zip(
                    sys_msg_meta_dicts, extend_sys_msg_meta_dicts
                )
            ]

        init_assistant_sys_msg, init_user_sys_msg = (
            sys_msg_generator.from_dicts(
                meta_dicts=sys_msg_meta_dicts,
                role_tuples=[
                    (assistant_role_name, RoleType.ASSISTANT),
                    (user_role_name, RoleType.USER),
                ],
            )
        )
        return init_assistant_sys_msg, init_user_sys_msg, sys_msg_meta_dicts

    def _init_agents(
        self,
        init_assistant_sys_msg: BaseMessage,
        init_user_sys_msg: BaseMessage,
        assistant_agent_kwargs: Optional[Dict] = None,
        user_agent_kwargs: Optional[Dict] = None,
        output_language: Optional[str] = None,
    ) -> None:
        r"""Initialize assistant and user agents with their system messages.

        Args:
            init_assistant_sys_msg (BaseMessage): Assistant agent's initial
                system message.
            init_user_sys_msg (BaseMessage): User agent's initial system
                message.
            assistant_agent_kwargs (Dict, optional): Additional arguments to
                pass to the assistant agent. (default: :obj:`None`)
            user_agent_kwargs (Dict, optional): Additional arguments to
                pass to the user agent. (default: :obj:`None`)
            output_language (str, optional): The language to be output by the
                agents. (default: :obj:`None`)
        """
        if self.model is not None:
            if assistant_agent_kwargs is None:
                assistant_agent_kwargs = {'model': self.model}
            elif 'model' not in assistant_agent_kwargs:
                assistant_agent_kwargs.update(dict(model=self.model))
            if user_agent_kwargs is None:
                user_agent_kwargs = {'model': self.model}
            elif 'model' not in user_agent_kwargs:
                user_agent_kwargs.update(dict(model=self.model))

        self.assistant_agent = ChatAgent(
            init_assistant_sys_msg,
            output_language=output_language,
            **(assistant_agent_kwargs or {}),
        )
        self.assistant_sys_msg = self.assistant_agent.system_message

        self.user_agent = ChatAgent(
            init_user_sys_msg,
            output_language=output_language,
            **(user_agent_kwargs or {}),
        )
        self.user_sys_msg = self.user_agent.system_message

    def _init_critic(
        self,
        sys_msg_generator: SystemMessageGenerator,
        sys_msg_meta_dicts: List[Dict],
        critic_role_name: str,
        critic_criteria: Optional[str] = None,
        critic_kwargs: Optional[Dict] = None,
    ) -> None:
        r"""Initialize critic agent. If critic role name is :obj:`"human"`,
        create a :obj:`Human` critic agent. Else, create a :obj:`CriticAgent`
        critic agent with specified critic criteria. If the critic criteria
        is not specified, set it to improve task performance.

        Args:
            sys_msg_generator (SystemMessageGenerator): A system message
                generator for agents.
            sys_msg_meta_dicts (list): A list of system message meta dicts.
            critic_role_name (str): The name of the role played by the critic.
            critic_criteria (str, optional): Critic criteria for the
                critic agent. If not specified, set the criteria to
                improve task performance. (default: :obj:`None`)
            critic_kwargs (Dict, optional): Additional arguments to
                pass to the critic. (default: :obj:`None`)
        """
        if self.with_critic_in_the_loop:
            if critic_role_name.lower() == "human":
                self.critic = Human(**(critic_kwargs or {}))
            else:
                critic_criteria = (
                    critic_criteria or "improving the task performance"
                )
                critic_msg_meta_dict = dict(
                    critic_role=critic_role_name,
                    criteria=critic_criteria,
                    **sys_msg_meta_dicts[0],
                )
                self.critic_sys_msg = sys_msg_generator.from_dict(
                    critic_msg_meta_dict,
                    role_tuple=(critic_role_name, RoleType.CRITIC),
                )
                if self.model is not None:
                    if critic_kwargs is None:
                        critic_kwargs = {'model': self.model}
                    elif 'model' not in critic_kwargs:
                        critic_kwargs.update(dict(model=self.model))
                self.critic = CriticAgent(
                    self.critic_sys_msg,
                    **(critic_kwargs or {}),
                )

    def _reduce_message_options(
        self,
        messages: Sequence[BaseMessage],
    ) -> BaseMessage:
        r"""Processes a sequence of chat messages, returning the processed
        message. If multiple messages are provided and
        `with_critic_in_the_loop` is `False`, raises a `ValueError`.
        If no messages are provided, a `ValueError` will be raised.

        Args:
            messages (Sequence[BaseMessage]): A sequence of `BaseMessage`
                objects to process.

        Returns:
            BaseMessage: A single `BaseMessage` representing the processed
                message.
        """
        if len(messages) == 0:
            raise ValueError("No messages to process.")
        if len(messages) > 1 and not self.with_critic_in_the_loop:
            raise ValueError(
                "Got than one message to process. "
                f"Num of messages: {len(messages)}."
            )
        elif self.with_critic_in_the_loop and self.critic is not None:
            critic_response = self.critic.reduce_step(messages)
            processed_msg = critic_response.msg
        else:
            processed_msg = messages[0]

        return processed_msg

    def init_chat(self, init_msg_content: Optional[str] = None) -> BaseMessage:
        r"""Initializes the chat by resetting both of the assistant and user
        agents. Returns an initial message for the role-playing session.

        Args:
            init_msg_content (str, optional): A user-specified initial message.
                Will be sent to the role-playing session as the initial
                message. (default: :obj:`None`)

        Returns:
            BaseMessage: A single `BaseMessage` representing the initial
                message.
        """
        self.assistant_agent.reset()
        self.user_agent.reset()
        default_init_msg_content = (
            "Now start to give me instructions one by one. "
            "Only reply with Instruction and Input."
        )
        if init_msg_content is None:
            init_msg_content = default_init_msg_content

        # Initialize a message sent by the assistant
        init_msg = BaseMessage.make_assistant_message(
            role_name=getattr(self.assistant_sys_msg, 'role_name', None)
            or "assistant",
            content=init_msg_content,
        )

        return init_msg

    async def ainit_chat(
        self, init_msg_content: Optional[str] = None
    ) -> BaseMessage:
        r"""Asynchronously initializes the chat by resetting both of the
        assistant and user agents. Returns an initial message for the
        role-playing session.

        Args:
            init_msg_content (str, optional): A user-specified initial message.
                Will be sent to the role-playing session as the initial
                message. (default: :obj:`None`)

        Returns:
            BaseMessage: A single `BaseMessage` representing the initial
                message.
        """
        # Currently, reset() is synchronous, but if it becomes async in the
        # future, we can await it here
        self.assistant_agent.reset()
        self.user_agent.reset()
        default_init_msg_content = (
            "Now start to give me instructions one by one. "
            "Only reply with Instruction and Input."
        )
        if init_msg_content is None:
            init_msg_content = default_init_msg_content

        # Initialize a message sent by the assistant
        init_msg = BaseMessage.make_assistant_message(
            role_name=getattr(self.assistant_sys_msg, 'role_name', None)
            or "assistant",
            content=init_msg_content,
        )

        return init_msg

    def step(
        self,
        assistant_msg: BaseMessage,
    ) -> Tuple[ChatAgentResponse, ChatAgentResponse]:
        r"""Advances the conversation by taking a message from the assistant,
        processing it using the user agent, and then processing the resulting
        message using the assistant agent. Returns a tuple containing the
        resulting assistant message, whether the assistant agent terminated
        the conversation, and any additional assistant information, as well as
        a tuple containing the resulting user message, whether the user agent
        terminated the conversation, and any additional user information.

        Args:
            assistant_msg: A `BaseMessage` representing the message from the
                assistant.

        Returns:
            Tuple[ChatAgentResponse, ChatAgentResponse]: A tuple containing two
                ChatAgentResponse: the first struct contains the resulting
                assistant message, whether the assistant agent terminated the
                conversation, and any additional assistant information; the
                second struct contains the resulting user message, whether the
                user agent terminated the conversation, and any additional user
                information.
        """
        user_response = self.user_agent.step(assistant_msg)
        if user_response.terminated or user_response.msgs is None:
            return (
                ChatAgentResponse(msgs=[], terminated=False, info={}),
                ChatAgentResponse(
                    msgs=[],
                    terminated=user_response.terminated,
                    info=user_response.info,
                ),
            )
        user_msg = self._reduce_message_options(user_response.msgs)

        # To prevent recording the same memory more than once (once in chat
        # step and once in role play), and the model generates only one
        # response when multi-response support is enabled.
        if (
            'n' in self.user_agent.model_backend.model_config_dict.keys()
            and self.user_agent.model_backend.model_config_dict['n'] > 1
        ):
            self.user_agent.record_message(user_msg)

        assistant_response = self.assistant_agent.step(user_msg)
        if assistant_response.terminated or assistant_response.msgs is None:
            return (
                ChatAgentResponse(
                    msgs=[],
                    terminated=assistant_response.terminated,
                    info=assistant_response.info,
                ),
                ChatAgentResponse(
                    msgs=[user_msg], terminated=False, info=user_response.info
                ),
            )
        assistant_msg = self._reduce_message_options(assistant_response.msgs)

        # To prevent recording the same memory more than once (once in chat
        # step and once in role play), and the model generates only one
        # response when multi-response support is enabled.
        if (
            'n' in self.assistant_agent.model_backend.model_config_dict.keys()
            and self.assistant_agent.model_backend.model_config_dict['n'] > 1
        ):
            self.assistant_agent.record_message(assistant_msg)

        return (
            ChatAgentResponse(
                msgs=[assistant_msg],
                terminated=assistant_response.terminated,
                info=assistant_response.info,
            ),
            ChatAgentResponse(
                msgs=[user_msg],
                terminated=user_response.terminated,
                info=user_response.info,
            ),
        )

    async def astep(
        self,
        assistant_msg: BaseMessage,
    ) -> Tuple[ChatAgentResponse, ChatAgentResponse]:
        r"""Asynchronously advances the conversation by taking a message from
        the assistant, processing it using the user agent, and then processing
        the resulting message using the assistant agent. Returns a tuple
        containing the resulting assistant message, whether the assistant
        agent terminated the conversation, and any additional assistant
        information, as well as a tuple containing the resulting user message,
        whether the user agent terminated the conversation, and any additional
        user information.

        Args:
            assistant_msg: A `BaseMessage` representing the message from the
                assistant.

        Returns:
            Tuple[ChatAgentResponse, ChatAgentResponse]: A tuple containing two
                ChatAgentResponse: the first struct contains the resulting
                assistant message, whether the assistant agent terminated the
                conversation, and any additional assistant information; the
                second struct contains the resulting user message, whether the
                user agent terminated the conversation, and any additional user
                information.
        """
        user_response = await self.user_agent.astep(assistant_msg)
        if user_response.terminated or user_response.msgs is None:
            return (
                ChatAgentResponse(msgs=[], terminated=False, info={}),
                ChatAgentResponse(
                    msgs=[],
                    terminated=user_response.terminated,
                    info=user_response.info,
                ),
            )
        user_msg = self._reduce_message_options(user_response.msgs)

        # To prevent recording the same memory more than once (once in chat
        # step and once in role play), and the model generates only one
        # response when multi-response support is enabled.
        if (
            'n' in self.user_agent.model_backend.model_config_dict.keys()
            and self.user_agent.model_backend.model_config_dict['n'] > 1
        ):
            self.user_agent.record_message(user_msg)

        assistant_response = await self.assistant_agent.astep(user_msg)
        if assistant_response.terminated or assistant_response.msgs is None:
            return (
                ChatAgentResponse(
                    msgs=[],
                    terminated=assistant_response.terminated,
                    info=assistant_response.info,
                ),
                ChatAgentResponse(
                    msgs=[user_msg], terminated=False, info=user_response.info
                ),
            )
        assistant_msg = self._reduce_message_options(assistant_response.msgs)

        # To prevent recording the same memory more than once (once in chat
        # step and once in role play), and the model generates only one
        # response when multi-response support is enabled.
        if (
            'n' in self.assistant_agent.model_backend.model_config_dict.keys()
            and self.assistant_agent.model_backend.model_config_dict['n'] > 1
        ):
            self.assistant_agent.record_message(assistant_msg)

        return (
            ChatAgentResponse(
                msgs=[assistant_msg],
                terminated=assistant_response.terminated,
                info=assistant_response.info,
            ),
            ChatAgentResponse(
                msgs=[user_msg],
                terminated=user_response.terminated,
                info=user_response.info,
            ),
        )
import json
import re
import uuid
from copy import deepcopy
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def merge_messages_with_tools(messages: list, tools: list) -> list:
    """
    Integrate tool information into the messages for open-sourced LLMs which don't support tool calling.

    Args:
        messages (list): A list of message dictionaries, each containing at least a "role" 
                         and "content" field. Some messages may contain "tool_calls".
        tools (list): A list of available tool definitions, formatted as dictionaries.

    Returns:
        list: The updated messages list, where:
              - Tool call messages are formatted properly for models without built-in tool support.
              - Messages indicating tool execution results are transformed into a user message.
              - The last message includes an instruction prompt detailing tool usage requirements.

    Example:
        ```python
        messages = [
            {"role": "user", "content": "Translate 'hello' to French."},
            {"role": "assistant", "tool_calls": [{"name": "translate", "parameters": {"text": "hello", "language": "fr"}}]}
        ]
        
        tools = [{"name": "translate", "description": "Translates text into another language."}]
        
        updated_messages = tool_calling_input_format(messages, tools)
        print(updated_messages)
        ```
    """
    tool_prompt = f"""
You have access to the following tools:

{json.dumps(tools)}"""
    
    format_prompt = f"""
To use a tool, respond with a JSON object in the following format:
```json
[{{"name": "tool_name","parameters": {{"arg1": "value1","arg2": "value2"}}}}]
```

Make sure your response is properly formatted as a valid JSON object.
"""
    
    new_messages = deepcopy(messages)

    new_messages[-1]["content"] += (tool_prompt + format_prompt)
    return new_messages

def merge_messages_with_response_format(messages: list, response_format: dict) -> list:
    """
    Format the response format instructions into a string for the prompt.
    
    Args:
        response_format: ResponseFormat object with schema information
        
    Returns:
        Formatted response format instructions as a string
    """
    if response_format:
        schema_str = json.dumps(response_format["json_schema"], indent=2)
        format_prompt = f"""
You MUST respond with a JSON object that conforms to the following schema:

{schema_str}

Your entire response must be valid JSON without any other text, preamble, or postscript.
Do not use code blocks like ```json or ```. Just return the raw JSON.
"""
    else:
        format_prompt = """
You MUST respond with a valid JSON object. 
Your entire response must be valid JSON without any other text, preamble, or postscript.
Do not use code blocks like ```json or ```. Just return the raw JSON.
"""

    new_messages = deepcopy(messages)

    new_messages[-1]["content"] += (format_prompt)
    return new_messages

def parse_json_format(message: str) -> str:
    """
    Extract and parse a JSON object or array from a given string.

    Args:
        message (str): The input string potentially containing a JSON object or array.

    Returns:
        str: A string representation of the extracted JSON object or array.
    
    Example:
        ```python
        message = "Here is some data: {\"key\": \"value\"}"
        parsed_json = parse_json_format(message)
        print(parsed_json)  # Output: '{"key": "value"}'
        ```
    """
    json_array_pattern = r"\[\s*\{.*?\}\s*\]"
    json_object_pattern = r"\{\s*.*?\s*\}"

    match_array = re.search(json_array_pattern, message)

    if match_array:
        json_array_substring = match_array.group(0)

        try:
            json_array_data = json.loads(json_array_substring)
            return json.dumps(json_array_data)
        except json.JSONDecodeError:
            pass

    match_object = re.search(json_object_pattern, message)

    if match_object:
        json_object_substring = match_object.group(0)

        try:
            json_object_data = json.loads(json_object_substring)
            return json.dumps(json_object_data)
        except json.JSONDecodeError:
            pass
    return "[]"

def decode_hf_tool_calls(message):
    """
    Decode tool call responses from Hugging Face API format.

    Args:
        message: The response object from Hugging Face API. 
    
    Returns:
        list: A list of dictionaries, each containing:
              - "name": The name of the function being called.
              - "parameters": The arguments passed to the function.
              - "id": The unique identifier of the tool call.
              
    Example:
        ```python
        response = <Hugging Face API response>
        decoded_calls = decode_hf_tool_calls(response)
        print(decoded_calls)  
        # Output: [{'name': 'translate', 'parameters': {'text': 'hello', 'lang': 'fr'}, 'id': 'uuid1234'}]  
        ```
    """
    message = message.replace("assistant\n\n", "")
    tool_calls = json.loads(message)
    for tool_call in tool_calls:
        tool_call["id"] = generator_tool_call_id()
    return tool_calls
    
def generator_tool_call_id():
    """
    Generate a unique identifier for a tool call.

    This function creates a new UUID (Universally Unique Identifier) and returns it as a string.

    Returns:
        str: A unique tool call ID.
    
    Example:
        ```python
        tool_call_id = generator_tool_call_id()
        print(tool_call_id)  # Example output: 'f3f2e850-b5d4-11ef-ac7e-96584d5248b2'
        ```
    """
    return str(uuid.uuid4())

def decode_litellm_tool_calls(response):
    """
    Decode tool call responses from LiteLLM/OpenAI style objects.

    Supports:
    - ModelResponse (response.choices[0].message.tool_calls)
    - A raw list of tool_call objects (ChatCompletionMessageFunctionToolCall)
    - (Fallback) JSON inside message.content
    """
    decoded_tool_calls = []

    # -----------------------------------
    # 0) ModelResponse 객체인지 먼저 확인
    # -----------------------------------
    tool_calls = None
    content = None

    # Case A: LiteLLM/OpenAI ModelResponse
    if hasattr(response, "choices"):
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        content = getattr(message, "content", None)

    # Case B: 이미 tool_calls 리스트를 직접 받은 경우
    elif isinstance(response, list) and response and hasattr(response[0], "function"):
        tool_calls = response
        content = None

    # -----------------------------------
    # 1) tool_calls 리스트가 있는 경우 (우선 경로)
    # -----------------------------------
    if tool_calls:
        for tool_call in tool_calls:
            # OpenAI / LiteLLM 스타일: tool_call.function.name, tool_call.function.arguments
            fn = getattr(tool_call, "function", None)
            name = None
            params = None

            if fn is not None:
                name = getattr(fn, "name", None)
                params = getattr(fn, "arguments", None)
            else:
                # 혹시 function이 아닌 다른 구조일 경우를 대비
                name = getattr(tool_call, "name", None)
                params = getattr(tool_call, "arguments", None)

            # arguments가 문자열 JSON이면 파싱
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except json.JSONDecodeError:
                    logger.info(
                        "decode_litellm_tool_calls: arguments is not valid JSON, keeping as raw string."
                    )

            if name is not None and params is not None:
                decoded_tool_calls.append(
                    {
                        "name": name,
                        "parameters": params,
                        "id": getattr(tool_call, "id", generator_tool_call_id()),
                    }
                )

        return decoded_tool_calls

    # -----------------------------------
    # 2) tool_calls가 없고, content 안에 JSON이 들어 있는 경우 (fallback)
    #    → 지금 vLLM + required 환경에서는 거의 안 쓸 가능성이 큼.
    # -----------------------------------
    if isinstance(content, str):
        tool_calls = content
        try:
            parsed = json.loads(tool_calls)
            if isinstance(parsed, (list, dict)):
                tool_calls = parsed
            else:
                logger.info(
                    "decode_litellm_tool_calls: unexpected JSON type for tool_calls: %s",
                    type(parsed),
                )
                tool_calls = []
        except json.JSONDecodeError:
            logger.info(
                "decode_litellm_tool_calls: non-JSON tool_calls string, treating as no tools."
            )
            tool_calls = []

        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]

        try:
            for tc in tool_calls:
                name = None
                if "name" in tc:
                    name = tc["name"]
                elif "function_name" in tc:
                    name = tc["function_name"]
                elif "tool_name" in tc:
                    name = tc["tool_name"]

                if name is not None:
                    parameters = None
                    if "arguments" in tc:
                        parameters = tc["arguments"]
                    elif "parameters" in tc:
                        parameters = tc["parameters"]

                    if parameters is not None:
                        decoded_tool_calls.append(
                            {
                                "name": name,
                                "parameters": parameters,
                                "id": generator_tool_call_id(),
                            }
                        )
        except Exception:
            logger.info(
                "decode_litellm_tool_calls: no valid attribute in tools, treating as no tools"
            )

    return decoded_tool_calls

def parse_tool_calls(message):
    """
    Parse and process tool calls from a message string.

    Args:
        message (str): A JSON string representing tool calls.

    Returns:
        list: A list of processed tool calls with unique IDs.

    Example:
        ```python
        message = '[{"name": "text_translate", "parameters": {"text": "hello", "lang": "fr"}}]'
        parsed_calls = parse_tool_calls(message)
        print(parsed_calls)  
        # Output: [{'name': 'text/translate', 'parameters': {'text': 'hello', 'lang': 'fr'}, 'id': 'uuid1234'}]
        ```
    """
    # add tool call id and type for models don't support tool call
    # if isinstance(message, dict):
    #     message = [message]
    # tool_calls = json.loads(parse_json_format(message))
    tool_calls = json.loads(message)
    # breakpoint()
    # tool_calls = json.loads(message)
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
        
    for tool_call in tool_calls:
        tool_call["id"] = generator_tool_call_id()
        # if "function" in tool_call:
        
    tool_calls = double_underscore_to_slash(tool_calls)
        # tool_call["type"] = "function"
    return tool_calls

def slash_to_double_underscore(tools):
    """
    Convert function names by replacing slashes ("/") with double underscores ("__").

    Args:
        tools (list): A list of tool dictionaries.

    Returns:
        list: The updated tools list with function names formatted properly.

    Example:
        ```python
        tools = [{"function": {"name": "text/translate"}}]
        formatted_tools = slash_to_double_underscore(tools)
        print(formatted_tools)  
        # Output: [{'function': {'name': 'text__translate'}}]
        ```
    """
    for tool in tools:
        tool_name = tool["function"]["name"]
        if "/" in tool_name:
            tool_name = "__".join(tool_name.split("/"))
            tool["function"]["name"] = tool_name
    return tools

def double_underscore_to_slash(tool_calls):
    """
    Convert function names by replacing double underscores ("__") back to slashes ("/").

    Args:
        tool_calls (list): A list of tool call dictionaries.

    Returns:
        list: The updated tool calls list with function names restored to their original format.

    Example:
        ```python
        tool_calls = [{"name": "text__translate", "parameters": '{"text": "hello", "lang": "fr"}'}]
        restored_calls = double_underscore_to_slash(tool_calls)
        print(restored_calls)  
        # Output: [{'name': 'text/translate', 'parameters': {'text': 'hello', 'lang': 'fr'}}]
        ```
    """
    for tool_call in tool_calls:
        tool_call["name"] = tool_call["name"].replace("__", "/")
        if isinstance(tool_call["parameters"], str):
            tool_call["parameters"] = json.loads(tool_call["parameters"])
        # tool_call["parameters"] = json.loads(tool_call["parameters"])
    return tool_calls

def pre_process_tools(tools):
    """
    Pre-process tool definitions by replacing slashes ("/") with double underscores ("__").

    Args:
        tools (list): A list of tool dictionaries.

    Returns:
        list: The processed tools list with modified function names.

    Example:
        ```python
        tools = [{"function": {"name": "text/translate"}}]
        preprocessed_tools = pre_process_tools(tools)
        print(preprocessed_tools)  
        # Output: [{'function': {'name': 'text__translate'}}]
        ```
    """
    for tool in tools:
        tool_name = tool["function"]["name"]
        if "/" in tool_name:
            tool_name = "__".join(tool_name.split("/"))
            tool["function"]["name"] = tool_name
    return tools

def check_availability_for_selected_llm_lists(available_llm_names: List[str], selected_llm_lists: List[List[Dict[str, Any]]]):
    selected_llm_lists_availability = []
    for selected_llm_list in selected_llm_lists:
        all_available = True
        
        for llm in selected_llm_list:
            if llm["name"] not in available_llm_names:
                all_available = False
                break
        selected_llm_lists_availability.append(all_available)
    return selected_llm_lists_availability

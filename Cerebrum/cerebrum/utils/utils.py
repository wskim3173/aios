import random
import os
from typing import Optional, Dict, Any
import re
import json

def generator_tool_call_id():
    """generate tool call id
    """
    return str(random.randint(0, 1000))

def get_from_env(env_key: str, default: Optional[str] = None) -> str:
    """Get a value from an environment variable."""
    if env_key in os.environ and os.environ[env_key]:
        return os.environ[env_key]
    elif default is not None:
        return default
    else:
        raise ValueError(
            f"Did not find {env_key}, please add an environment variable"
            f" `{env_key}` which contains it. "
        )

'''
def _parse_json_output(text: str) -> Dict[str, Any]:
    r"""Extract JSON output from a string."""

    markdown_pattern = r'```(?:json)?\s*(.*?)\s*```'
    markdown_match = re.search(markdown_pattern, text, re.DOTALL)
    if markdown_match:
        text = markdown_match.group(1).strip()

    triple_quotes_pattern = r'"""(?:json)?\s*(.*?)\s*"""'
    triple_quotes_match = re.search(triple_quotes_pattern, text, re.DOTALL)
    if triple_quotes_match:
        text = triple_quotes_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            fixed_text = re.sub(
                r'`([^`]*?)`(?=\s*[:,\[\]{}]|$)', r'"\1"', text
            )
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            result = {}
            try:
                bool_pattern = r'"(\w+)"\s*:\s*(true|false)'
                for match in re.finditer(bool_pattern, text, re.IGNORECASE):
                    key, value = match.groups()
                    result[key] = value.lower() == "true"

                str_pattern = r'"(\w+)"\s*:\s*"([^"]*)"'
                for match in re.finditer(str_pattern, text):
                    key, value = match.groups()
                    result[key] = value

                num_pattern = r'"(\w+)"\s*:\s*(-?\d+(?:\.\d+)?)'
                for match in re.finditer(num_pattern, text):
                    key, value = match.groups()
                    try:
                        result[key] = int(value)
                    except ValueError:
                        result[key] = float(value)

                empty_str_pattern = r'"(\w+)"\s*:\s*""'
                for match in re.finditer(empty_str_pattern, text):
                    key = match.group(1)
                    result[key] = ""

                if result:
                    return result

                print(f"Failed to parse JSON output: {text}")
                return {}
            except Exception as e:
                print(f"Error while extracting fields from JSON: {e}")
                return {}
'''#kws

def _parse_json_output(text: Any) -> Dict[str, Any]:
    """Extract a JSON object from various LLM outputs.
    - Accepts str / dict / None
    - Handles fenced blocks (```json ... ```, ``` ... ```), triple quotes
    - Tries raw JSON first, then a light "backtick→quote" fix, then braces fallback
    """
    # 0) Normalize to string
    if text is None:
        return {}
    if not isinstance(text, str):
        # Common chat formats: try to dig out textual content first
        if isinstance(text, dict):
            for key in ("content", "message", "text"):
                if isinstance(text.get(key), str):
                    text = text[key]
                    break
            else:
                # As a last resort, stringify dict
                try:
                    text = json.dumps(text, ensure_ascii=False)
                except Exception:
                    text = str(text)
        else:
            try:
                text = json.dumps(text, ensure_ascii=False)
            except Exception:
                text = str(text)

    # 1) Trim fenced code blocks: ```json ... ``` or ``` ... ```
    md = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if md:
        text = md.group(1).strip()

    # 2) Trim triple quotes """json ... """
    tq = re.search(r'"""(?:json)?\s*(.*?)\s*"""', text, re.DOTALL)
    if tq:
        text = tq.group(1).strip()

    # 3) Try direct JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # 4) Light fix: inline backticks around keys/values → quotes
    try:
        fixed_text = re.sub(r'`([^`]*?)`(?=\s*[:,\[\]{}]|$)', r'"\1"', text)
        return json.loads(fixed_text)
    except Exception:
        pass

    # 5) Fallback: extract the first {...} block and parse
    try:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            return json.loads(brace.group(0))
    except Exception:
        pass

    # 6) Very last fallback: heuristic field extractor (best-effort)
    result: Dict[str, Any] = {}
    try:
        # booleans
        for m in re.finditer(r'"(\w+)"\s*:\s*(true|false)', text, re.IGNORECASE):
            k, v = m.groups()
            result[k] = v.lower() == "true"
        # strings
        for m in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', text):
            k, v = m.groups()
            result[k] = v
        # numbers
        for m in re.finditer(r'"(\w+)"\s*:\s*(-?\d+(?:\.\d+)?)', text):
            k, v = m.groups()
            result[k] = int(v) if re.fullmatch(r"-?\d+", v) else float(v)
        return result if result else {}
    except Exception:
        return {}
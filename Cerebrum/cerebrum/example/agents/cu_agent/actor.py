from typing import Dict, Tuple, Optional
from .mcp_client import MCPClient

class Actor:
    def __init__(self, mcp_client: MCPClient):
        self.mcp_client = mcp_client
        
    async def act(self, action_dict: Dict) -> Tuple[bool, str]:
        """Execute an action based on the given action code.
        
        Args:
            action_dict (Dict): The action to execute with parameters
            
        Returns:
            Tuple[bool, str]: (success, info) indicating if the action was successful
        """
        action_type = action_dict.get("action_type", "")
        parameters = action_dict.get("parameters", {})
        
        action_result = await self.mcp_client.call_tool(action_type, parameters)
        return action_result
        
    async def get_screenshot(self) -> bytes:
        """Get the current screen state.
        
        Returns:
            bytes: The screenshot data
        """
        screenshot_response = await self.mcp_client.call_tool("screenshot", {})
        return screenshot_response.content[0].text
        
    async def get_accessibility_tree(self) -> str:
        """Get the current accessibility tree.
        
        Returns:
            str: The accessibility tree data
        """
        accessibility_tree_response = await self.mcp_client.call_tool("get_accessibility_tree", {})
        return accessibility_tree_response.content[0].text
        
    async def start_recording(self):
        """Start recording the session."""
        await self.mcp_client.call_tool("start_recording", {})
        
    async def stop_recording(self, file_name: str):
        """Stop recording and save to file.
        
        Args:
            file_name (str): Path to save the recording
        """
        await self.mcp_client.call_tool("stop_recording", {"file_name": file_name})
        
    async def evaluate(self, action_history: list) -> float:
        """Evaluate the task performance.
        
        Args:
            action_history (list): History of actions taken
            
        Returns:
            float: Evaluation score
        """
        try:
            score = await self.mcp_client.call_tool("evaluate", {"action_history": action_history})
            return float(score.content[0].text)
        except Exception as e:
            print(f"Error evaluating the task: {e}")
            return 0.0 
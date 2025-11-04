import unittest
import os
import sys
import json
import time
import re
from pathlib import Path
import logging
import subprocess
from typing import Dict, List, Any

from cerebrum.manager.agent import AgentManager
from cerebrum.config.config_manager import config
from cerebrum.commands.run_agent import AgentConfig, AgentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent_tests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestAgentsWithOllama(unittest.TestCase):
    """Test all available agents using ollama/qwen2.5:7b as the LLM."""
    
    @classmethod
    def setUpClass(cls):
        """Set up the test environment once before all tests."""
        # Check if ollama is installed and running
        cls._check_ollama_status()
        
        # Get agent hub URL from config
        cls.agent_hub_url = config.get_agent_hub_url()
        
        # Initialize agent manager
        cls.agent_manager = AgentManager(cls.agent_hub_url)
        
        # Get all available agents (both from hub and local)
        cls.hub_agents = cls.agent_manager.list_agenthub_agents()
        cls.local_agents = cls.agent_manager.list_local_agents()
        
        logger.info(f"Found {len(cls.hub_agents)} agents in AgentHub")
        logger.info(f"Found {len(cls.local_agents)} local agents")
        
        # Create results directory
        cls.results_dir = Path("test_results")
        cls.results_dir.mkdir(exist_ok=True)
        
        # Define test tasks for different agent types
        cls.agent_tasks = cls._prepare_agent_tasks()
    
    @classmethod
    def _check_ollama_status(cls):
        """Check if ollama is installed and running, and if qwen2.5:7b is available."""
        try:
            # Check if ollama is installed
            subprocess.run(["ollama", "--version"], check=True, capture_output=True)
            
            # Check if ollama server is running
            subprocess.run(["curl", "-s", "http://localhost:11434/api/tags"], check=True, capture_output=True)
            
            # Check if qwen2.5:7b is available
            result = subprocess.run(
                ["ollama", "list"], 
                check=True, 
                capture_output=True, 
                text=True
            )
            
            if "qwen2.5:7b" not in result.stdout:
                logger.warning("qwen2.5:7b model not found in ollama. Attempting to pull...")
                subprocess.run(["ollama", "pull", "qwen2.5:7b"], check=True)
                
            logger.info("Ollama is installed, running, and qwen2.5:7b model is available")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Ollama check failed: {e}")
            logger.error("Please ensure ollama is installed, running, and qwen2.5:7b model is available")
            raise unittest.SkipTest("Ollama prerequisites not met")
    
    @classmethod
    def _prepare_agent_tasks(cls) -> Dict[str, Dict[str, str]]:
        """Prepare specific tasks for different agent types based on their names or descriptions."""
        return {
            # General agents
            "default": {
                "task": "Tell me briefly about artificial intelligence.",
                "description": "General task for any agent"
            },
            # Academic agents
            "academic": {
                "task": "Summarize the key findings of recent papers about large language models.",
                "description": "Academic research task"
            },
            # Math agents
            "math": {
                "task": "Solve the equation: 3x^2 + 5x - 2 = 0",
                "description": "Mathematical problem solving task"
            },
            # Travel agents
            "travel": {
                "task": "Suggest a 3-day itinerary for visiting Tokyo, Japan.",
                "description": "Travel planning task"
            },
            # Recommendation agents
            "recommend": {
                "task": "Recommend three science fiction books similar to 'Dune'.",
                "description": "Recommendation task"
            },
            # Creative agents
            "creative": {
                "task": "Write a short poem about technology and nature.",
                "description": "Creative writing task"
            },
            # Code agents
            "code": {
                "task": "Write a Python function to check if a string is a palindrome.",
                "description": "Coding task"
            },
            # Test agents
            "test": {
                "task": "Hello, can you introduce yourself and your capabilities?",
                "description": "Simple introduction task"
            }
        }
    
    def _get_appropriate_task(self, agent: Dict[str, Any]) -> str:
        """Determine the appropriate task for an agent based on its name and description."""
        name = agent.get("name", "").lower()
        description = agent.get("description", "").lower() if isinstance(agent.get("description"), str) else ""
        
        # Check for specific agent types in name or description
        for agent_type, task_info in self.agent_tasks.items():
            if agent_type == "default":
                continue
                
            if agent_type in name or agent_type in description:
                logger.info(f"Using {agent_type} specific task for agent {name}")
                return task_info["task"]
        
        # Default task if no specific type is matched
        logger.info(f"Using default task for agent {name}")
        return self.agent_tasks["default"]["task"]
    
    def _validate_response(self, response: str, agent_id: str = "") -> bool:
        """
        Validate that the response is meaningful and doesn't contain error messages.
        
        Args:
            response: The agent's response
            agent_id: The agent identifier for logging
            
        Returns:
            bool: True if the response is valid, False otherwise
        """
        # Check if response is empty or too short
        if not response or len(response.strip()) < 10:
            logger.error(f"Agent {agent_id} returned empty or too short response")
            return False
        
        # Check for common error patterns
        error_patterns = [
            r"(?i)error occurred",
            r"(?i)exception",
            r"(?i)failed to",
            r"(?i)unable to",
            r"(?i)could not",
            r"(?i)I don't have the capability",
            r"(?i)I cannot",
            r"(?i)I'm not able to",
            r"(?i)I am unable to",
            r"(?i)I don't have access",
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, response):
                # Check if it's a false positive (e.g., explaining how to handle errors)
                if "how to handle errors" in response.lower() or "error handling" in response.lower():
                    continue
                    
                logger.warning(f"Agent {agent_id} response contains potential error: {pattern}")
                logger.warning(f"Response excerpt: {response[:100]}...")
                return False
        
        return True
    
    def _run_agent_test(self, agent: Dict[str, Any], is_local: bool = False) -> Dict[str, Any]:
        """
        Run a test for a single agent and return the result.
        
        Args:
            agent: The agent information dictionary
            is_local: Whether the agent is local or from AgentHub
            
        Returns:
            Dict[str, Any]: Test result information
        """
        agent_name = agent.get("name", "Unknown")
        agent_id = f"{agent.get('author', 'local')}/{agent_name}"
        
        logger.info(f"Testing {'local' if is_local else 'hub'} agent: {agent_id}")
        
        try:
            # Get appropriate task for this agent
            task = self._get_appropriate_task(agent)
            
            # Record start time
            start_time = time.time()
            
            if is_local:
                # For local agents, use the path
                agent_path = agent.get("path")
                if not agent_path:
                    raise ValueError(f"No path found for local agent {agent_name}")
                
                agent_config = AgentConfig(
                    agent_path=agent_path,
                    task_input=task,
                    agenthub_url=self.agent_hub_url,
                    mode="local"
                )
            else:
                # For hub agents, use author, name, version
                agent_config = AgentConfig(
                    agent_author=agent.get("author"),
                    agent_name=agent_name,
                    agent_version=agent.get("version"),
                    task_input=task,
                    agenthub_url=self.agent_hub_url,
                    mode="remote"
                )
            
            # Run the agent
            agent_runner = AgentRunner(agent_config)
            response = agent_runner.run()["result"]
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Validate response
            is_valid = self._validate_response(response, agent_id)
            
            if is_valid:
                status = "success"
                logger.info(f"✅ Agent {agent_id} test passed in {execution_time:.2f} seconds")
            else:
                status = "invalid_response"
                logger.error(f"❌ Agent {agent_id} returned invalid response")
            
            # Return test result
            return {
                "agent_id": agent_id,
                "status": status,
                "execution_time": execution_time,
                "task": task,
                "response": response,
                "is_valid": is_valid
            }
            
        except Exception as e:
            logger.error(f"❌ Agent {agent_id} test failed: {str(e)}")
            return {
                "agent_id": agent_id,
                "status": "failed",
                "error": str(e),
                "is_valid": False
            }
    
    def test_all_agents_from_agenthub(self):
        """Test all available agents from AgentHub with appropriate tasks."""
        # Skip test if no agents found
        if not self.hub_agents:
            self.skipTest("No agents found in AgentHub to test")
        
        results = {}
        success_count = 0
        failure_count = 0
        
        # Limit to first 3 agents for faster testing
        test_agents = self.hub_agents[:3]
        
        for agent in test_agents:
            result = self._run_agent_test(agent, is_local=False)
            
            # Track results
            results[result["agent_id"]] = result
            if result.get("is_valid", False):
                success_count += 1
            else:
                failure_count += 1
            
            # Add a small delay between tests
            time.sleep(1)
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        result_file = self.results_dir / f"agenthub_test_results_{timestamp}.json"
        
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"AgentHub test results saved to {result_file}")
        logger.info(f"Tests completed: {success_count} succeeded, {failure_count} failed")
        
        # Assert that all agents succeeded
        self.assertEqual(success_count, len(test_agents), "All tested AgentHub agents should succeed")

    def test_all_agents_from_local_folder(self):
        """Test all agents from local folder."""
        # Skip test if no local agents found
        if not self.local_agents:
            self.skipTest("No local agents found to test")
            
        self.local_agents = self.local_agents[:3]
        
        results = {}
        success_count = 0
        failure_count = 0
        
        for agent in self.local_agents:
            result = self._run_agent_test(agent, is_local=True)
            
            # Track results
            results[result["agent_id"]] = result
            if result.get("is_valid", False):
                success_count += 1
            else:
                failure_count += 1
            
            # Add a small delay between tests
            time.sleep(1)
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        result_file = self.results_dir / f"local_agent_test_results_{timestamp}.json"
        
        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Local agent test results saved to {result_file}")
        logger.info(f"Tests completed: {success_count} succeeded, {failure_count} failed")
        
        # Assert that all agents succeeded
        self.assertEqual(success_count, len(self.local_agents), "All local agents should succeed")

if __name__ == "__main__":
    # Run only the local agents test
    suite = unittest.TestSuite()
    suite.addTest(TestAgentsWithOllama('test_all_agents_from_local_folder'))
    unittest.TextTestRunner().run(suite)

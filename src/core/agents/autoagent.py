from typing import Dict, Any
from ...models.model_manager import model_manager
from ...core.utils import setup_logger

logger = setup_logger(__name__)

class AutoAgent:
    def __init__(self, model_name: str, task_description: str):
        self.model_name = model_name
        self.task_description = task_description
        self.context: Dict[str, Any] = {}

    def process_input(self, user_input: str) -> str:
        try:
            prompt = f"Task: {self.task_description}\nContext: {self.context}\nUser Input: {user_input}\nAgent:"
            self._update_context(user_input, prompt)
            return prompt
        except Exception as e:
            logger.error(f"Error processing input for AutoAgent: {str(e)}")
            raise

    def _update_context(self, user_input: str, agent_response: str) -> None:
        # Implement context update logic here
        # This could involve summarization, key information extraction, etc.
        self.context["last_user_input"] = user_input
        self.context["last_agent_response"] = agent_response

class AutoAgentFactory:
    @staticmethod
    def create_agent(model_name: str, task_description: str) -> AutoAgent:
        try:
            model_manager.load_model(model_name)  # Ensure the model is loaded
            return AutoAgent(model_name, task_description)
        except Exception as e:
            logger.error(f"Error creating AutoAgent: {str(e)}")
            raise

auto_agent_factory = AutoAgentFactory()
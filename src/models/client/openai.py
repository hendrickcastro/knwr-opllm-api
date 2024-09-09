from openai import OpenAI
from typing import Any, Dict, Optional, List
from ...contract.IClient import IClient
from ...core.config import settings
from ...core.utils import setup_logger
from ...core.common.functions import ToolFunctions

logger = setup_logger(__name__)

class OpenAIModel(IClient):
    def __init__(self, modelName: str):
        self.modelName = modelName
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def load(self) -> None:
        # OpenAI models don't need to be explicitly loaded
        pass

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> Optional[object]:
        if (settings.DEBUGG):
            for key, value in kwargs.items():
                print(f"{key}: {value}")
                
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            
            # Convert the response to a dictionary
            response_dict = {
                "message": {"content": response.choices[0].message.content},
                "done_reason": response.choices[0].finish_reason,
                "done": True,
                "total_duration": response.usage.total_tokens,  # This is not exactly duration, but a close approximation
                "prompt_eval_count": response.usage.prompt_tokens,
                "eval_count": response.usage.completion_tokens,
            }
            return response_dict
        except Exception:
            raise
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            
            response_dict = {
                "message": {"content": response.choices[0].message.content},
                "done_reason": response.choices[0].finish_reason,
                "done": True,
                "total_duration": response.usage.total_tokens,
                "prompt_eval_count": response.usage.prompt_tokens,
                "eval_count": response.usage.completion_tokens,
            }
            
            return response_dict
        except Exception as e:
            logger.error(f"Error generating text with OpenAI model {self.modelName}: {str(e)}")
            raise
        
    def _filter_kwargs(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        accepted_params = [
            'model', 'temperature', 'top_p', 'n', 'stream', 'stop', 'max_tokens',
            'presence_penalty', 'frequency_penalty', 'logit_bias', 'user',
            'response_format', 'seed', 'tools', 'tool_choice'
        ]
        return {k: v for k, v in kwargs.items() if k in accepted_params and v is not None}

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.modelName, "type": "openai"}

    def generate_embedding(self, text: str) -> str:
        try:
            response = self.client.embeddings.create(
                model=self.modelName,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding with OpenAI model {self.modelName}: {str(e)}")
            raise

    def create_chunks(self, content: str, content_type: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def process_auto_agent(self, user_input: str, task_description: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def get_models(self) -> List[Dict[str, str]]:
        try:
            response = self.client.models.list()
            return [{"id": model.id, "object": model.object} for model in response.data]
        except Exception as e:
            logger.error(f"Error getting models from OpenAI: {str(e)}")
            raise

    def generate_prompt(self, prompt: str) -> str:
        # Placeholder implementation as details are needed
        return prompt

    def generate_prompts(self, messages: List[Dict[str, str]]) -> str:
        # Placeholder implementation as details are needed
        return " ".join([message["content"] for message in messages])
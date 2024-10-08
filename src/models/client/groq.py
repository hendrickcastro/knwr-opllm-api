from typing import Any, Dict, Optional, List
from ...contract.IClient import IClient
from ...core.config import settings
from ...core.utils import setup_logger
from groq import Groq
from ...core.common.functions import ToolFunctions

logger = setup_logger(__name__)

class GroqModel(IClient):
    def __init__(self, modelName: str):
        self.modelName = modelName
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def load(self) -> None:
        # Groq models don't need to be explicitly loaded
        pass

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = 500, temperature: float = 0.7, **kwargs) -> Optional[object]:
        try:
            response = self.client.chat.completions.create(
                model=self.modelName,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
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
            raise e
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> Optional[object]:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.modelName,
                messages= messages,
                max_tokens=max_tokens,
                temperature=temperature
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
            logger.error(f"Error generating text with Groq model {self.modelName}: {str(e)}")
            raise
        
    def _filter_kwargs(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        accepted_params = [
            'max_tokens', 'temperature', 'top_p', 'stream',
            'stop', 'presence_penalty', 'frequency_penalty'
        ]
        return {k: v for k, v in kwargs.items() if k in accepted_params and v is not None}

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.modelName, "type": "groq"}

    def generate_embedding(self, text: str) -> str:
        try:
            response = self.client.embeddings.create(
                model=self.modelName,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding with Groq model {self.modelName}: {str(e)}")
            raise

    def create_chunks(self, content: str, content_type: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def process_auto_agent(self, user_input: str, task_description: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def get_models(self) -> List[Dict[str, str]]:
        try:
            # Note: Groq might not have an API endpoint to list models
            # This is a placeholder implementation
            return [{"id": self.modelName, "object": "model"}]
        except Exception as e:
            logger.error(f"Error getting models from Groq: {str(e)}")
            raise

    def generate_prompt(self, prompt: str) -> str:
        # Placeholder implementation as details are needed
        return prompt

    def generate_prompts(self, messages: List[Dict[str, str]]) -> str:
        # Placeholder implementation as details are needed
        return " ".join([message["content"] for message in messages])
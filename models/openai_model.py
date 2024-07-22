from openai import OpenAI
from typing import Any, Dict, Optional, List
from .base_model import BaseModel
from core.config import settings
from core.utils import setup_logger

logger = setup_logger(__name__)

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def load(self) -> None:
        # OpenAI models don't need to be explicitly loaded
        pass

    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> str:
        try:
            messages = self._format_messages([{"role": "user", "content": prompt}])
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating with OpenAI model {self.model_name}: {str(e)}")
            raise

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> str:
        try:
            formatted_messages = self._format_messages(messages)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating chat with OpenAI model {self.model_name}: {str(e)}")
            raise

    def _format_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        formatted_messages = []
        for message in messages:
            formatted_message = {
                "role": message["role"],
                "content": [
                    {
                        "type": "text",
                        "text": message["content"]
                    }
                ]
            }
            formatted_messages.append(formatted_message)
        return formatted_messages

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "openai"}
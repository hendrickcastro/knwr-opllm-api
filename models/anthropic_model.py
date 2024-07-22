from typing import Any, Dict, Optional, List
from anthropic import Anthropic
from .base_model import BaseModel
from core.utils import setup_logger
from core.config import settings

logger = setup_logger(__name__)

class AnthropicModel(BaseModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def load(self) -> None:
        # No es necesario cargar explícitamente
        pass

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or 1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating with Anthropic model {self.model_name}: {str(e)}")
            raise

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or 1024,
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating chat with Anthropic model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "anthropic"}

# Asegúrate de que la clase esté siendo exportada
__all__ = ['AnthropicModel']
import asyncio
from typing import Any, Dict, Optional, List
from xai_sdk import Client
from ...core.config import settings
from ...contract.IClient import IClient
import logging

logger = logging.getLogger(__name__)

class GrokModel(IClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = Client(api_key=settings.GROK_API_KEY)

    def load(self) -> None:
        # No se necesita cargar explícitamente para Grok
        pass

    async def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> str:
        try:
            response_text = ""
            async for token in self.client.sampler.sample(prompt, max_len=max_tokens or 100):
                response_text += token.token_str
            return response_text
        except Exception as e:
            logger.error(f"Error generating with Grok model {self.model_name}: {str(e)}")
            raise

    async def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> str:
        try:
            prompt = self._create_chat_prompt(messages)
            return await self.generate(prompt, max_tokens, temperature, **kwargs)
        except Exception as e:
            logger.error(f"Error generating chat with Grok model {self.model_name}: {str(e)}")
            raise

    def _create_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        for message in messages:
            role = message['role'].capitalize()
            content = message['content']
            prompt += f"{role}: {content}\n\n"
        prompt += "Assistant: "
        return prompt

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "grok"}

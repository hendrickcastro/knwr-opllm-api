from openai import OpenAI
from typing import Any, Dict, Optional, List
from ..base_model import BaseModel
from ...core.config import settings
from ...core.utils import setup_logger

logger = setup_logger(__name__)

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def load(self) -> None:
        # OpenAI models don't need to be explicitly loaded
        pass

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> str:
        
        ## impruve 
        #   "top_p": 0.9,
        #   "n": 1,
        #   "stop": ["\n", " user:"],
        #    "presence_penalty": 0.6,
        #   "frequency_penalty": 0.0,
        #   "user": "example_user"
        
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error generating chat with OpenAI model {self.model_name}: {str(e)}")
            raise
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating text with OpenAI model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "openai"}
import requests
import json
from typing import Any, Dict, Optional, List
from ...contract.IClient import IClient
from ...core.config import settings
from ...core.utils import setup_logger

logger = setup_logger(__name__)

class OllamaModel(IClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = settings.OLLAMA_BASE_URL

    def load(self) -> None:
        if not self._is_model_available():
            raise ValueError(f"Model {self.model_name} is not available in Ollama")

    def _is_model_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return any(self.model_name in model for model in available_models)
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama model availability: {str(e)}")
            return False

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> str:
        try:
            formatted_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            formatted_messages += "\nassistant:"
            
            payload = {
                "model": self.model_name,
                "prompt": formatted_messages
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if temperature is not None:
                payload["temperature"] = temperature
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            if response.status_code == 200:
                response_text = response.text
                response_lines = response_text.strip().split('\n')
                full_response = ""
                for line in response_lines:
                    try:
                        response_json = json.loads(line)
                        full_response += response_json.get("response", "")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line as JSON: {line}")
                return full_response.strip()
            else:
                raise Exception(f"Ollama API error: {response.text}")
        except Exception as e:
            logger.error(f"Error generating chat with Ollama model {self.model_name}: {str(e)}")
            raise
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
            payload["temperature"] = temperature
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            if response.status_code == 200:
                response_text = response.text
                response_lines = response_text.strip().split('\n')
                full_response = ""
                for line in response_lines:
                    try:
                        response_json = json.loads(line)
                        full_response += response_json.get("response", "")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line as JSON: {line}")
                return full_response.strip()
            else:
                raise Exception(f"Ollama API error: {response.text}")
        except Exception as e:
            logger.error(f"Error generating with Ollama model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "ollama"}
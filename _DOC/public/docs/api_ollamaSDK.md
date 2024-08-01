
## Archivo: ollamaSDK.py
### Ruta Relativa: ../src\models\client\ollamaSDK.py

```python
import json
import ollama
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
            logger.info(f"Model {self.model_name} is not available. Attempting to load it.")
            if not self._load_model():
                raise ValueError(f"Model {self.model_name} could not be loaded in Ollama")

    def _is_model_available(self) -> bool:
        try:
            response = ollama.get_models()
            available_models = [model["name"] for model in response]
            return self.model_name in available_models
        except AttributeError:
            logger.error("The method 'get_models' does not exist in the ollama module")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama model availability: {str(e)}")
            return False

    def _load_model(self) -> bool:
        try:
            response = ollama.load_model(self.model_name)
            if response.get('status') == 'success':
                return True
            else:
                logger.error(f"Failed to load model {self.model_name}: {response.get('message', 'Unknown error')}")
                return False
        except AttributeError:
            logger.error("The method 'load_model' does not exist in the ollama module")
            return False
        except Exception as e:
            logger.error(f"Error loading Ollama model {self.model_name}: {str(e)}")
            return False

    @staticmethod
    def get_available_models() -> List[str]:
        try:
            response = ollama.get_models()
            available_models = [model["name"] for model in response]
            return available_models
        except AttributeError:
            logger.error("The method 'get_models' does not exist in the ollama module")
            return []
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")
            return []

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> str:
        try:
            response = ollama.chat(model=self.model_name, messages=messages)
            if 'message' in response:
                response_text = response['message']
                full_response = ""
                for line in response_text.split('\n'):
                    try:
                        response_json = json.loads(line)
                        full_response += response_json.get("response", "")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line as JSON: {line}")
                return full_response.strip()
            else:
                raise Exception(f"Ollama API error: {response}")
        except Exception as e:
            logger.error(f"Error generating chat with Ollama model {self.model_name}: {str(e)}")
            raise
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        try:
            response = ollama.generate(model=self.model_name, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            if 'message' in response:
                response_text = response['message']
                full_response = ""
                for line in response_text.split('\n'):
                    try:
                        response_json = json.loads(line)
                        full_response += response_json.get("response", "")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line as JSON: {line}")
                return full_response.strip()
            else:
                raise Exception(f"Ollama API error: {response}")
        except Exception as e:
            logger.error(f"Error generating with Ollama model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "ollama"}
```

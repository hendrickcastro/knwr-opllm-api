
## Archivo: ollama.py
### Ruta Relativa: ../src\models\client\ollama.py

```python
import json
import ollama
from typing import Any, Dict, Optional, List
from ...contract.IClient import IClient
from ...core.config import settings
from ...core.utils import setup_logger
from ...core.common.functions import ToolFunctions

logger = setup_logger(__name__)

class OllamaModel(IClient):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def load(self) -> None:
        # Assuming Ollama models don't need explicit loading
        pass

    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
            )
            return response['text'].strip()
        except Exception as e:
            logger.error(f"Error generating text with Ollama model {self.model_name}: {str(e)}")
            raise

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> Optional[object]:
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={ **kwargs, temperature: temperature }
            )

            return response
        except Exception as e:
            logger.error(f"Error generating chat with Ollama model {self.model_name}: {str(e)}")
            raise
        
    def _filter_kwargs(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        accepted_params = [ "num_keep", "seed", "num_predict", "top_k", "top_p", "min_p", "tfs_z", "typical_p", "repeat_last_n",
        "temperature", "repeat_penalty", "presence_penalty", "frequency_penalty", "mirostat", "mirostat_tau", "mirostat_eta",
        "penalize_newline", "stop", "numa", "num_ctx", "num_batch", "num_gpu", "main_gpu", "low_vram", "f16_kv", "vocab_only",
        "use_mmap", "use_mlock", "num_thread" ]
        return {k: v for k, v in kwargs.items() if k in accepted_params and v is not None}

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "ollama"}

    def generate_embedding(self, text: str) -> str:
        try:
            response = ollama.embedding(
                model=self.model_name,
                input=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding with Ollama model {self.model_name}: {str(e)}")
            raise

    def create_chunks(self, content: str, content_type: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def process_auto_agent(self, user_input: str, task_description: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def get_models(self) -> List[Dict[str, str]]:
        try:
            response = ollama.list_models()
            return [{"id": model['id'], "object": model['object']} for model in response['models']]
        except Exception as e:
            logger.error(f"Error getting models from Ollama: {str(e)}")
            raise

    def generate_prompt(self, prompt: str) -> str:
        # Placeholder implementation as details are needed
        return prompt

    def generate_prompts(self, messages: List[Dict[str, str]]) -> str:
        # Placeholder implementation as details are needed
        return " ".join([message["content"] for message in messages])
```


## Archivo: ollama_prompt_handler.py
### Ruta Relativa: ../api\prompts\ollama_prompt_handler.py

```python
from typing import List, Dict
from .base_prompt_handler import BasePromptHandler

class OllamaPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        formatted_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return f"{formatted_messages}\nassistant:"

```

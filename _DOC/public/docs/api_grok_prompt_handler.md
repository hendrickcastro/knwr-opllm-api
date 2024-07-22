
## Archivo: grok_prompt_handler.py
### Ruta Relativa: ../api\prompts\grok_prompt_handler.py

```python
from typing import List, Dict
from .base_prompt_handler import BasePromptHandler

class GrokPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

```

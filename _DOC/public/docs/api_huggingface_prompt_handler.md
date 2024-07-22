
## Archivo: huggingface_prompt_handler.py
### Ruta Relativa: ../api\prompts\huggingface_prompt_handler.py

```python
from typing import List, Dict
from .base_prompt_handler import BasePromptHandler

class HuggingFacePromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        return " ".join([msg['content'] for msg in messages])

```

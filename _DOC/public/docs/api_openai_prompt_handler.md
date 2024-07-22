
## Archivo: openai_prompt_handler.py
### Ruta Relativa: ../api\prompts\openai_prompt_handler.py

```python
from typing import List, Dict
from .base_prompt_handler import BasePromptHandler

class OpenAIPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return messages

```


## Archivo: huggingface_prompt.py
### Ruta Relativa: ../src\models\prompts\huggingface_prompt.py

```python
from typing import List, Dict
from .base_prompt import BasePromptHandler

class HuggingFacePromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        return " ".join([msg['content'] for msg in messages])

```

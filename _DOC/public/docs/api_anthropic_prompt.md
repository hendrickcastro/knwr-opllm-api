
## Archivo: anthropic_prompt.py
### Ruta Relativa: ../src\models\prompts\anthropic_prompt.py

```python
from typing import List, Dict, Any
from .base_prompt import BasePromptHandler

class AnthropicPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        system_message = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
        chat_messages = [msg for msg in messages if msg['role'] != 'system']
        return {
            "system": system_message,
            "messages": chat_messages
        }

```

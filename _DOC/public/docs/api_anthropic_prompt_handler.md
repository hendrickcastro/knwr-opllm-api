
## Archivo: anthropic_prompt_handler.py
### Ruta Relativa: ../api\prompts\anthropic_prompt_handler.py

```python
from typing import List, Dict, Any
from .base_prompt_handler import BasePromptHandler

class AnthropicPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        system_message = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
        chat_messages = [msg for msg in messages if msg['role'] != 'system']
        return {
            "system": system_message,
            "messages": chat_messages
        }

```

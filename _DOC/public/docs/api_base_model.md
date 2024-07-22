
## Archivo: base_model.py
### Ruta Relativa: ../api\models\base_model.py

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

class BaseModel(ABC):
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        pass

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        # Default implementation for models that don't support chat natively
        prompt = self._create_chat_prompt(messages)
        return self.generate(prompt, max_tokens)

    def _create_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        for message in messages:
            role = message['role'].capitalize()
            content = message['content']
            prompt += f"{role}: {content}\n\n"
        prompt += "Assistant: "
        return prompt
```

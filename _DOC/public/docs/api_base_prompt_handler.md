
## Archivo: base_prompt_handler.py
### Ruta Relativa: ../api\prompts\base_prompt_handler.py

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BasePromptHandler(ABC):
    @abstractmethod
    def format_prompt(self, messages: List[Dict[str, str]]) -> Any:
        pass

```


## Archivo: base_prompt.py
### Ruta Relativa: ../src\models\prompts\base_prompt.py

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BasePromptHandler(ABC):
    @abstractmethod
    def format_prompt(self, messages: List[Dict[str, str]]) -> Any:
        pass

```

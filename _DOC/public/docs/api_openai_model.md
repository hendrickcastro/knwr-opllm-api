
## Archivo: openai_model.py
### Ruta Relativa: ../api\models\openai_model.py

```python
from openai import OpenAI
from typing import Any, Dict, Optional
from .base_model import BaseModel
from core.config import settings
from core.utils import setup_logger

logger = setup_logger(__name__)

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def load(self) -> None:
        # OpenAI models don't need to be explicitly loaded
        pass

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        try:
            # Determinar el tipo de modelo y el endpoint a utilizar
            if 'gpt-4o-mini' in self.model_name or 'gpt-3.5-turbo' in self.model_name or 'gpt-4' in self.model_name:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens or 100
                )
                return response.choices[0].message.content.strip()
            else:
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens or 100
                )
                return response.choices[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating with OpenAI model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "openai"}
```

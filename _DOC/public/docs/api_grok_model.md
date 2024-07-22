
## Archivo: grok_model.py
### Ruta Relativa: ../api\models\grok_model.py

```python
from typing import Any, Dict, Optional
import requests
from .base_model import BaseModel
from core.utils import setup_logger
from core.config import settings

logger = setup_logger(__name__)

class GrokModel(BaseModel):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = "https://api.grok.com"  # URL base de la API de Grok
        self.headers = {
            "Authorization": f"Bearer {settings.GROK_API_KEY}",
            "Content-Type": "application/json"
        }

    def load(self) -> None:
        # Aquí puedes agregar lógica para verificar si el modelo está disponible en Grok si es necesario
        pass

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens or 100
            }
            
            logger.info(f"Sending payload to Grok API: {payload}")
            response = requests.post(f"{self.base_url}/generate", json=payload, headers=self.headers)
            logger.info(f"Grok API response status code: {response.status_code}")
            logger.info(f"Grok API response content: {response.text}")

            if response.status_code == 200:
                return response.json()["generated_text"]
            else:
                raise Exception(f"Grok API error: {response.text}")
        except Exception as e:
            logger.error(f"Error generating with Grok model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "grok"}

```

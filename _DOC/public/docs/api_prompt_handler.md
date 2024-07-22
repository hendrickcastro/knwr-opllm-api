
## Archivo: prompt_handler.py
### Ruta Relativa: ../api\prompts\prompt_handler.py

```python
from pydantic import BaseModel
from models.model_manager import model_manager
from core.utils import setup_logger
from .prompt_handler_factory import PromptHandlerFactory

logger = setup_logger(__name__)

class Message(BaseModel):
    role: str
    content: str

class PromptHandler:
    def __init__(self):
        self.model_manager = model_manager

    def process_prompt(self, model_name: str, prompt_type: str, **kwargs) -> str:
        try:
            model = self.model_manager.get_model(model_name)
            model_type = model.get_info()['type']
            handler = PromptHandlerFactory.get_handler(model_type)
            
            if prompt_type == "chat":
                messages = kwargs.get('messages', [])
                formatted_messages = [msg.dict() for msg in messages]
                formatted_prompt = handler.format_prompt(formatted_messages)
            else:
                prompt = kwargs.get('input', '')
                formatted_prompt = handler.format_prompt([{"role": "user", "content": prompt}])
            
            return model.generate(formatted_prompt)
        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}")
            raise

prompt_handler = PromptHandler()

```

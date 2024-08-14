from typing import List, Dict
from pydantic import BaseModel
from models.model_manager import model_manager
from core.utils import setup_logger

logger = setup_logger(__name__)

class Message(BaseModel):
    role: str
    content: str

class PromptHandler:
    def __init__(self):
        self.model_manager = model_manager

    def process_prompt(self, modelName: str, prompt_type: str, **kwargs) -> str:
        try:
            model = self.model_manager.get_model(modelName)
            
            if prompt_type == "chat":
                messages = kwargs.get('messages', [])
                # Convertir los objetos Message a diccionarios
                dict_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
                return model.generate_chat(dict_messages)
            else:
                prompt = kwargs.get('input', '')
                return model.generate(prompt)
        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}")
            raise

prompt_handler = PromptHandler()
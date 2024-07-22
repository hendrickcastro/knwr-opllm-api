from pydantic import BaseModel
from models.model_manager import model_manager
from core.utils import setup_logger
from .prompt_handler_factory import PromptHandlerFactory
from typing import List, Optional

logger = setup_logger(__name__)

class Message(BaseModel):
    role: str
    content: str

class PromptHandler:
    def __init__(self):
        self.model_manager = model_manager

    def process_prompt(self, model_name: str, prompt_type: str, messages: List[Message], max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        try:
            model = self.model_manager.get_model(model_name)
            model_type = model.get_info()['type']
            handler = PromptHandlerFactory.get_handler(model_type)
            
            formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            if model_type == "openai":
                # Para modelos de OpenAI, pasamos los mensajes directamente
                return model.generate(formatted_messages, max_tokens=max_tokens, temperature=temperature)
            else:
                # Para otros tipos de modelos, usamos el handler para formatear el prompt
                formatted_prompt = handler.format_prompt(formatted_messages)
                return model.generate(formatted_prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}")
            raise

prompt_handler = PromptHandler()
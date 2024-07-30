from typing import Any, Dict, Optional, List
from anthropic import Anthropic
from ...contract.IClient import IClient
from ...core.utils import setup_logger
from ...core.config import settings
import json

logger = setup_logger(__name__)

class AnthropicModel(IClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def load(self) -> None:
        # No es necesario cargar explícitamente para Anthropic
        pass

    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating with Anthropic model {self.model_name}: {str(e)}")
            raise

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> Optional[object]:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                messages=messages,
                **kwargs
            )
            # Convertimos la respuesta a un diccionario
            resp = {
                "message": {"content": response.content[0].text},
                "done_reason": response.stop_reason,
                "done": True,
                "total_duration": None,  # Anthropic no proporciona esta información
                "load_duration": None,
                "prompt_eval_count": None,
                "prompt_eval_duration": None,
                "eval_count": None,
                "eval_duration": None,
            }
            
            return resp
        except Exception as e:
            logger.error(f"Error generating chat with Anthropic model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "anthropic"}

    def generate_embedding(self, text: str) -> str:
        # Nota: Anthropic actualmente no ofrece un servicio de embeddings.
        # Esta es una implementación ficticia y debería ser reemplazada cuando esté disponible.
        logger.warning("Anthropic no ofrece servicio de embeddings. Retornando texto original.")
        return text

    def create_chunks(self, content: str, content_type: str) -> str:
        chunks = []
        for i in range(0, len(content), 1024):
            chunk = content[i:i+1024]
            if content_type == 'json':
                chunks.append(json.dumps({"role": "user", "content": chunk}))
            else:
                chunks.append(chunk)
        return json.dumps(chunks)  # Retornamos una cadena JSON

    def process_auto_agent(self, user_input: str, task_description: str) -> str:
        # Nota: Anthropic no tiene un método específico para esto.
        # Esta es una implementación aproximada usando el método de chat.
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": task_description},
                    {"role": "user", "content": user_input}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error in process_auto_agent with Anthropic model {self.model_name}: {str(e)}")
            raise

    def get_models(self) -> List[Dict[str, str]]:
        # Anthropic no proporciona una API para listar modelos.
        # Retornamos una lista estática de modelos conocidos.
        return [
            {"name": "claude-2", "type": "chat"},
            {"name": "claude-instant-1", "type": "chat"}
        ]

    def generate_prompt(self, prompt: str) -> str:
        # Anthropic no requiere un procesamiento especial del prompt
        return prompt

    def generate_prompts(self, messages: List[Dict[str, str]]) -> str:
        # Convertimos los mensajes al formato esperado por Anthropic
        return json.dumps([{"role": msg.get("role", "user"), "content": msg["content"]} for msg in messages])

# Asegúrate de que la clase esté siendo exportada
__all__ = ['AnthropicModel']
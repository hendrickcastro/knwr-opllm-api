from typing import Any, Dict, Optional, List
from anthropic import Anthropic
from ...contract.IClient import IClient
from ...core.utils import setup_logger
from ...core.config import settings
from ...core.common.functions import ToolFunctions
import json

logger = setup_logger(__name__)

class AnthropicModel(IClient):
    def __init__(self, modelName: str):
        self.modelName = modelName
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def load(self) -> None:
        # No es necesario cargar explícitamente para Anthropic
        pass

    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        try:
            response = self.client.messages.create(
                model=self.modelName,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return self._format_response(response)
        except Exception as e:
            logger.error(f"Error generating with Anthropic model {self.modelName}: {str(e)}")
            raise

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        try:
            system_message = ""
            filtered_messages = []
            
            for message in messages:
                if message["role"].lower() == "system":
                    system_message += message["content"] + " "
                else:
                    filtered_messages.append(message)
            
            system_message = system_message.strip()
            
            response = self.client.messages.create(
                model=self.modelName,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                system=system_message if system_message else None,
                messages=filtered_messages,
                **kwargs
            )
            resp = self._format_response(response)
                
            return resp
        except Exception as e:
            logger.error(f"Error generating chat with Anthropic model {self.modelName}: {str(e)}")
            raise

    def _format_response(self, response) -> Dict[str, Any]:
        content = response.content
        # Si content es una lista, tomamos el primer elemento
        if isinstance(content, list) and len(content) > 0:
            content = content[0].text if hasattr(content[0], 'text') else str(content[0])
        else:
            content = str(content)

        return {
            "message": { "role": "user", "content": content },
            "tool_calls": [],
            "done_reason": response.stop_reason,
            "done": True,
            "total_duration": None,
            "load_duration": None,
            "prompt_eval_count": response.usage.input_tokens if response.usage else None,
            "prompt_eval_duration": None,
            "eval_count": response.usage.output_tokens if response.usage else None,
            "eval_duration": None,
        }
        
    def _filter_kwargs(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        # Lista de parámetros aceptados por Anthropic
        accepted_params = ['temperature', 'top_p', 'top_k', 'max_tokens', 'stream', 'metadata']
        return {k: v for k, v in kwargs.items() if k in accepted_params and v is not None}

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.modelName, "type": "anthropic"}

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
                model=self.modelName,
                messages=[
                    {"role": "system", "content": task_description},
                    {"role": "user", "content": user_input}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error in process_auto_agent with Anthropic model {self.modelName}: {str(e)}")
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
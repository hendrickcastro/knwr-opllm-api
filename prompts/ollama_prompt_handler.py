from typing import List, Dict, Any
from .base_prompt_handler import BasePromptHandler

class OllamaPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        formatted_messages = []
        for message in messages:
            if isinstance(message, dict) and 'role' in message and 'content' in message:
                formatted_message = {
                    "role": message["role"],
                    "content": [
                        {
                            "type": "text",
                            "text": message["content"]
                        }
                    ]
                }
                formatted_messages.append(formatted_message)
            else:
                raise ValueError("Each message must be a dictionary with 'role' and 'content' keys")
        return formatted_messages

    def format_completion(self, message: str) -> List[Dict[str, Any]]:
        if isinstance(message, str):
            lprompt = [{"role": "user", "content": message}]
            return self.format_prompt(lprompt)
        else:
            raise TypeError("Expected a string for 'message'")

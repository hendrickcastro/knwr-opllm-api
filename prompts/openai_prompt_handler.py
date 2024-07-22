from typing import List, Dict, Any
from .base_prompt_handler import BasePromptHandler

class OpenAIPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        formatted_messages = []
        for message in messages:
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
        return formatted_messages
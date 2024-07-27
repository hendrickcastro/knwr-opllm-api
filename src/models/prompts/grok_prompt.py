from typing import List, Dict
from .base_prompt import BasePromptHandler

class GrokPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

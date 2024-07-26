from typing import List, Dict
from ..base_prompt import BasePromptHandler

class HuggingFacePromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        return " ".join([msg['content'] for msg in messages])

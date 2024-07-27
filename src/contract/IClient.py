from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

class IClient(ABC):
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        pass

    @abstractmethod
    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> str:
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def generate_embedding(self, text: str) -> str:
        pass
    
    @abstractmethod
    def create_chunks(self, content: str, content_type: str) -> str:
        pass
    
    @abstractmethod
    def process_auto_agent(self, user_input: str, task_description: str) -> str:
        pass
    
    @abstractmethod
    def get_models(self) -> List[Dict[str, str]]:
        pass
    
    @abstractmethod
    def generate_prompt(self, prompt: str) -> str:
        pass
    
    @abstractmethod
    def generate_prompts(self, messages: List[Dict[str, str]]) -> str:
        pass

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from pydantic import Field
from pydantic import BaseModel as PBaseModel, Field

class BaseModel(ABC):
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> str:
        # Default implementation for models that don't support chat natively
        
        # model = self.model_manager.get_model(model_name)
        # model_type = model.get_info()['type']
        # handler = PromptHandlerFactory.get_handler(model_type)
            
        # prompt = self._create_chat_prompt(messages)
        # return self.generate(prompt, max_tokens, temperature
        pass


    # @abstractmethod
    # def _create_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
    #     # prompt = ""
    #     # for message in messages:
    #     #     role = message['role'].capitalize()
    #     #     content = message['content']
    #     #     prompt += f"{role}: {content}\n\n"
    #     # prompt += "Assistant: "
    #     # return prompt
    #     pass
    
    
class Message(PBaseModel):
    role: str
    content: str

class ChatRequest(PBaseModel):
    model_name: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: float = 0.7

class ChatResponse(PBaseModel):
    response: str

class EmbeddingRequest(PBaseModel):
    text: str

    class Config:
        protected_namespaces = ()

class EmbeddingResponse(PBaseModel):
    embedding: List[float]

class ChunkRequest(PBaseModel):
    content: str
    content_type: str

    class Config:
        protected_namespaces = ()

class ChunkResponse(PBaseModel):
    chunks: List[str]

class AutoAgentRequest(PBaseModel):
    model_name: str = Field(..., alias='model_name')
    task_description: str
    user_input: str

    class Config:
        protected_namespaces = ()

class AutoAgentResponse(PBaseModel):
    response: str

class GenerateRequest(PBaseModel):
    model: str = Field(..., alias='model')
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None

class CompareEmbeddingsRequest(PBaseModel):
    text1: str
    text2: str

class CompareEmbeddingsResponse(PBaseModel):
    similarity: float

class StoreEmbeddingRequest(PBaseModel):
    text: str
    metadata: Dict[str, Any]

class StoreEmbeddingResponse(PBaseModel):
    embedding_id: str

class SearchSimilarEmbeddingsRequest(PBaseModel):
    text: str
    top_k: int = 5

class SimilarEmbedding(PBaseModel):
    id: str
    metadata: Dict[str, Any]
    cosine_similarity: float

class SearchSimilarEmbeddingsResponse(PBaseModel):
    similar_embeddings: List[SimilarEmbedding]

class GenerateResponse(PBaseModel):
    generated_text: str
    
class RAGRequest(PBaseModel):
    query: str
    model_name: str
    top_k: int = 5

class RAGResponse(PBaseModel):
    answer: str
    sources: List[Dict[str, Any]]
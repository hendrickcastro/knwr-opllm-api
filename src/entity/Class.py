from typing import Any, Dict, Optional, List
from pydantic import Field
from pydantic import BaseModel as PBaseModel, Field

class Message(PBaseModel):
    role: str
    content: str

class ChatRequest(PBaseModel):
    model_name: str
    messages: List[Message]
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: Optional[float] = None
    n: Optional[int] = None
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None
    stream: Optional[bool] = False
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    best_of: Optional[int] = None

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
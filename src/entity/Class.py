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
    # stream: Optional[bool] = False
    # logit_bias: Optional[Dict[str, float]] = None
    # logprobs: Optional[int] = None
    # echo: Optional[bool] = False
    # best_of: Optional[int] = None

class ChatResponse(PBaseModel):
    message: str
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    done_reason: Optional[str] = None
    done: Optional[bool] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

    @classmethod
    def from_model_response(cls, response: Dict[str, Any]) -> 'ChatResponse':
        if isinstance(response.get("message"), dict):
            message_content = response["message"].get("content", "")
        elif isinstance(response.get("message"), list):
            # Si es una lista, tomamos el primer elemento
            message_content = response["message"][0].text if hasattr(response["message"][0], 'text') else str(response["message"][0])
        else:
            message_content = str(response.get("message", ""))

        return cls(
            message=message_content,
            tool_calls=response.get("tool_calls", []),
            done_reason=response.get("done_reason"),
            done=response.get("done"),
            total_duration=response.get("total_duration"),
            load_duration=response.get("load_duration"),
            prompt_eval_count=response.get("prompt_eval_count"),
            prompt_eval_duration=response.get("prompt_eval_duration"),
            eval_count=response.get("eval_count"),
            eval_duration=response.get("eval_duration")
        )

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
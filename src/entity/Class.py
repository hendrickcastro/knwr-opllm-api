from typing import Any, Dict, Optional, List, Union
from pydantic import Field
from pydantic import BaseModel as PBaseModel, Field

class Message(PBaseModel):
    role: str
    content: str
    
class Session(PBaseModel):
    userId: Optional[str] = None
    sessionId: Optional[str] = None
    
class RequestBasic(PBaseModel):
    prompt: Optional[str] = None
    modelName: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    session: Optional[Session] = None

class ChatRequest(PBaseModel):
    # Parámetros obligatorios
    modelName: str
    messages: List[Message]
    session: Optional[Session] = None

    # Parámetros comunes opcionales
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stream: Optional[bool] = None
    stop: Optional[Union[str, List[str]]] = None

    # Parámetros específicos de OpenAI
    n: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    response_format: Optional[Dict[str, str]] = None
    seed: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # Parámetros específicos de Hugging Face
    min_length: Optional[int] = None
    do_sample: Optional[bool] = None
    early_stopping: Optional[bool] = None
    num_beams: Optional[int] = None
    repetition_penalty: Optional[float] = None
    bad_words_ids: Optional[List[List[int]]] = None
    bos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    length_penalty: Optional[float] = None
    no_repeat_ngram_size: Optional[int] = None
    encoder_no_repeat_ngram_size: Optional[int] = None
    num_return_sequences: Optional[int] = None
    max_time: Optional[float] = None
    max_new_tokens: Optional[int] = None
    decoder_start_token_id: Optional[int] = None
    use_cache: Optional[bool] = None
    num_beam_groups: Optional[int] = None
    diversity_penalty: Optional[float] = None
    prefix_allowed_tokens_fn: Optional[Any] = None
    output_attentions: Optional[bool] = None
    output_hidden_states: Optional[bool] = None
    output_scores: Optional[bool] = None
    return_dict_in_generate: Optional[bool] = None
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    remove_invalid_values: Optional[bool] = None
    exponential_decay_length_penalty: Optional[tuple] = None
    suppress_tokens: Optional[List[int]] = None
    begin_suppress_tokens: Optional[List[int]] = None
    forced_decoder_ids: Optional[List[List[int]]] = None
    sequence_bias: Optional[Dict[str, float]] = None
    guidance_scale: Optional[float] = None
    low_memory: Optional[bool] = None

    # Parámetros específicos de Ollama
    num_keep: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    tfs_z: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = None
    temperature: Optional[float] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    penalize_newline: Optional[bool] = None
    stop: Optional[List[str]] = None
    numa: Optional[bool] = None
    num_ctx: Optional[int] = None
    num_batch: Optional[int] = None
    num_gpu: Optional[int] = None
    main_gpu: Optional[int] = None
    low_vram: Optional[bool] = None
    f16_kv: Optional[bool] = None
    vocab_only: Optional[bool] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    num_thread: Optional[int] = None

    # Puedes agregar más parámetros específicos de otros modelos si es necesario

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
        elif isinstance(response.get("response"), str):
            message_content = response.get("response", "")
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
    modelName: str = Field(..., alias='modelName')
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
    session: Optional[Session] = None

class StoreEmbeddingResponse(PBaseModel):
    embedding_id: str

class SearchSimilarEmbeddingsRequest(PBaseModel):
    text: str
    top_k: int = 5
    session: Optional[Session] = None
    cosine_similarity: Optional[float] = 0.6

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
    modelName: str
    top_k: int = 5
    session: Optional[Session] = None

class RAGResponse(PBaseModel):
    answer: str
    sources: List[SimilarEmbedding]
    
class ProcessFileResponse(PBaseModel):
    filename: str
    total_chunks: int
    embedding_ids: List[str]
    
class EmbeddingResponse(PBaseModel):
    embedding: List[float]

class SimilarEmbedding(PBaseModel):
    id: str
    metadata: Dict[str, Any]
    cosine_similarity: Optional[float] = None
    content: Optional[str] = None

class ListEmbeddingsResponse(PBaseModel):
    embeddings: List[SimilarEmbedding]

class GetEmbeddingResponse(PBaseModel):
    embedding: SimilarEmbedding
    
class SyncResponse(PBaseModel):
    message: str
    
class Query(PBaseModel):
    columns: List[str] = []
    where: Dict[str, Any] = {}
    order_by: List[str] = []
    limit: Optional[int] = None
    offset: Optional[int] = None
    
    
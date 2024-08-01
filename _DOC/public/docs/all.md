
## Archivo: router.py
### Ruta Relativa: ../src\api\routes\router.py

```python
from fastapi import APIRouter, HTTPException
from typing import List, Dict
from ...core.agents.autoagent import auto_agent_factory
from ...models.model_manager import model_manager
from ...entity.Class import ChatResponse, ChatRequest, AutoAgentRequest, AutoAgentResponse, GenerateRequest, GenerateResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        logger.info(f"Received generate request: {request.dict()}")
        kwargs = {}

        kwargs['temperature'] = request.temperature
        generated_text = model_manager.generate(
            request.model,
            request.prompt,
            max_tokens=request.max_tokens,
            **kwargs
        )
        return GenerateResponse(generated_text=generated_text)
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request for model: {request.model_name}")
        kwargs = request.dict(exclude={"model_name", "messages", "max_tokens", "temperature"})
        
        response = model_manager.generate_chat(
            model_name=request.model_name,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            **kwargs
        )
        
        return ChatResponse.from_model_response(response)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/autoagent", response_model=AutoAgentResponse)
async def process_auto_agent(request: AutoAgentRequest):
    try:
        agent = auto_agent_factory.create_agent(request.model_name, request.task_description)
        response = agent.process_input(request.user_input)
        return AutoAgentResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```


## Archivo: router_check.py
### Ruta Relativa: ../src\api\routes\router_check.py

```python
from fastapi import APIRouter, HTTPException
from typing import List, Dict
from ...models.model_manager import model_manager
import logging

logger = logging.getLogger(__name__)

router_check = APIRouter()

@router_check.get("/models", response_model=List[Dict[str, str]])
async def list_models():
    try:
        return model_manager.list_loaded_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_check.post("/load_model")
async def load_model(model_name: str, model_type: str):
    try:
        model_manager.load_model(model_name, model_type)
        return {"message": f"Model {model_name} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router_check.get("/check_ollama")
async def check_ollama():
    try:
        return model_manager.check_ollama_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_check.get("/check_ollama_models")
async def check_ollama_models():
    try:
        return model_manager.list_ollama_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```


## Archivo: router_storage.py
### Ruta Relativa: ../src\api\routes\router_storage.py

```python
from fastapi import APIRouter, HTTPException
from ...models.embeddings import embedding_generator
from ...core.storage.database import db
from ...models.model_manager import model_manager
from ...core.chunks.chunk_handler import chunk_handler
from ...entity.Class import CompareEmbeddingsRequest, EmbeddingRequest, EmbeddingResponse, ChunkRequest, ChunkResponse, CompareEmbeddingsResponse, StoreEmbeddingRequest, StoreEmbeddingResponse, SearchSimilarEmbeddingsRequest, SearchSimilarEmbeddingsResponse, RAGRequest, RAGResponse, SimilarEmbedding
import logging
import traceback

logger = logging.getLogger(__name__)

router_storage = APIRouter()

@router_storage.post("/embedding", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    try:
        embedding = embedding_generator.generate_embedding(request.text)
        return EmbeddingResponse(embedding=embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_storage.post("/chunk", response_model=ChunkResponse)
async def create_chunks(request: ChunkRequest):
    try:
        chunks = chunk_handler.process_chunks(request.content, request.content_type)
        return ChunkResponse(chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_storage.post("/compare_embeddings", response_model=CompareEmbeddingsResponse)
async def compare_embeddings(request: CompareEmbeddingsRequest):
    try:
        embedding1 = embedding_generator.generate_embedding(request.text1)
        embedding2 = embedding_generator.generate_embedding(request.text2)
        similarity = embedding_generator.compare_embeddings(embedding1, embedding2)
        return CompareEmbeddingsResponse(similarity=similarity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_storage.post("/store_embedding", response_model=StoreEmbeddingResponse)
async def store_embedding(request: StoreEmbeddingRequest):
    try:
        embedding = embedding_generator.generate_embedding(request.text)
        embedding_id = db.store_embedding(embedding, request.metadata)
        return StoreEmbeddingResponse(embedding_id=embedding_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_storage.post("/search_similar_embeddings", response_model=SearchSimilarEmbeddingsResponse)
async def search_similar_embeddings(request: SearchSimilarEmbeddingsRequest):
    try:
        logger.info(f"Received request to search similar embeddings for text: {request.text[:50]}...")
        query_embedding = embedding_generator.generate_embedding(request.text)
        logger.info(f"Generated embedding of length {len(query_embedding)}")
        similar_embeddings = db.search_similar_embeddings(query_embedding, request.top_k)
        logger.info(f"Found {len(similar_embeddings)} similar embeddings")
        
        formatted_embeddings = []
        for embedding in similar_embeddings:
            try:
                formatted_embedding = SimilarEmbedding(
                    id=str(embedding['_id']),
                    metadata=embedding['metadata'],
                    cosine_similarity=float(embedding['cosine_similarity'])
                )
                formatted_embeddings.append(formatted_embedding)
                logger.info(f"Formatted embedding: {formatted_embedding}")
            except KeyError as ke:
                logger.error(f"KeyError while formatting embedding: {ke}")
                logger.error(f"Problematic embedding: {embedding}")
            except Exception as e:
                logger.error(f"Error formatting embedding: {str(e)}")
                logger.error(f"Problematic embedding: {embedding}")
        
        return SearchSimilarEmbeddingsResponse(similar_embeddings=formatted_embeddings)
    except Exception as e:
        logger.error(f"Error in search_similar_embeddings endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@router_storage.post("/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    try:
        # 1. Generate embedding for the query
        query_embedding = embedding_generator.generate_embedding(request.query)
        
        # 2. Search for similar embeddings in the database
        similar_embeddings = db.search_similar_embeddings(query_embedding, request.top_k)
        
        # 3. Retrieve the content associated with these embeddings
        context = "\n".join([embedding['metadata']['content'] for embedding in similar_embeddings])
        
        # 4. Generate a prompt that includes the context and the query
        prompt = f"Context:\n{context}\n\nQuestion: {request.query}\n\nAnswer:"
        
        # 5. Use the language model to generate an answer
        answer = model_manager.generate(request.model_name, prompt)
        
        # 6. Return the answer and the sources
        sources = [{"id": str(emb['_id']), "metadata": emb['metadata']} for emb in similar_embeddings]
        return RAGResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
```


## Archivo: IClient.py
### Ruta Relativa: ../src\contract\IClient.py

```python
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
    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> Optional[object]:
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

```


## Archivo: autoagent.py
### Ruta Relativa: ../src\core\agents\autoagent.py

```python
from typing import Dict, Any
from ...models.model_manager import model_manager
from ...models.prompts.prompt_handler import prompt_handler
from ...core.utils import setup_logger

logger = setup_logger(__name__)

class AutoAgent:
    def __init__(self, model_name: str, task_description: str):
        self.model_name = model_name
        self.task_description = task_description
        self.context: Dict[str, Any] = {}

    def process_input(self, user_input: str) -> str:
        try:
            prompt = f"Task: {self.task_description}\nContext: {self.context}\nUser Input: {user_input}\nAgent:"
            response = prompt_handler.process_prompt(self.model_name, "completion", input=prompt)
            self._update_context(user_input, response)
            return response
        except Exception as e:
            logger.error(f"Error processing input for AutoAgent: {str(e)}")
            raise

    def _update_context(self, user_input: str, agent_response: str) -> None:
        # Implement context update logic here
        # This could involve summarization, key information extraction, etc.
        self.context["last_user_input"] = user_input
        self.context["last_agent_response"] = agent_response

class AutoAgentFactory:
    @staticmethod
    def create_agent(model_name: str, task_description: str) -> AutoAgent:
        try:
            model_manager.load_model(model_name)  # Ensure the model is loaded
            return AutoAgent(model_name, task_description)
        except Exception as e:
            logger.error(f"Error creating AutoAgent: {str(e)}")
            raise

auto_agent_factory = AutoAgentFactory()
```


## Archivo: chunk_handler.py
### Ruta Relativa: ../src\core\chunks\chunk_handler.py

```python
from typing import List, Union
from .text_chunks import text_chunker
from .code_chunks import code_chunker
from ..utils import setup_logger

logger = setup_logger(__name__)

class ChunkHandler:
    def __init__(self):
        self.text_chunker = text_chunker
        self.code_chunker = code_chunker

    def process_chunks(self, content: str, content_type: str) -> List[str]:
        try:
            if content_type == 'text':
                return self.text_chunker.chunk_text(content)
            elif content_type == 'code':
                return self.code_chunker.chunk_code(content)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
        except Exception as e:
            logger.error(f"Error processing chunks: {str(e)}")
            raise

    def set_text_chunk_size(self, size: int) -> None:
        self.text_chunker.set_chunk_size(size)

    def set_text_overlap(self, overlap: int) -> None:
        self.text_chunker.set_overlap(overlap)

    def set_code_max_lines(self, max_lines: int) -> None:
        self.code_chunker.set_max_lines(max_lines)

chunk_handler = ChunkHandler()
```


## Archivo: code_chunks.py
### Ruta Relativa: ../src\core\chunks\code_chunks.py

```python
import re
from typing import List
from ..utils import setup_logger

logger = setup_logger(__name__)

class CodeChunker:
    def __init__(self, max_lines: int = 50):
        self.max_lines = max_lines

    def chunk_code(self, code: str, language: str) -> List[str]:
        try:
            lines = code.split('\n')
            chunks = []
            current_chunk = []

            for line in lines:
                current_chunk.append(line)
                if len(current_chunk) >= self.max_lines or self._is_chunk_boundary(line, language):
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []

            if current_chunk:
                chunks.append('\n'.join(current_chunk))

            logger.info(f"Code chunked into {len(chunks)} parts")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking code: {str(e)}")
            raise

    def _is_chunk_boundary(self, line: str, language: str) -> bool:
        patterns = {
            "python": [
                r'^\s*def\s+\w+\s*\(.*\):',  # Function definition
                r'^\s*class\s+\w+.*:',       # Class definition
                r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]:'  # Main block
            ],
            "javascript": [
                r'^\s*function\s+\w+\s*\(.*\)\s*{',  # Function definition
                r'^\s*class\s+\w+\s*{',              # Class definition
                r'^\s*const\s+\w+\s*=\s*function\s*\(.*\)\s*{',  # Arrow function
                r'^\s*export\s+',                    # Export statement
                r'^\s*import\s+'                     # Import statement
            ],
            "csharp": [
                r'^\s*public\s+(class|interface|struct|enum)\s+\w+',  # Class, interface, struct, or enum
                r'^\s*(public|private|protected)\s+\w+\s+\w+\s*\(.*\)',  # Method definition
                r'^\s*namespace\s+',                 # Namespace
                r'^\s*using\s+'                      # Using statement
            ]
        }
        
        return any(re.match(pattern, line) for pattern in patterns.get(language, []))

    def set_max_lines(self, max_lines: int) -> None:
        self.max_lines = max_lines
        logger.info(f"Max lines per chunk set to {max_lines}")

code_chunker = CodeChunker()
```


## Archivo: text_chunks.py
### Ruta Relativa: ../src\core\chunks\text_chunks.py

```python
from typing import List
from ..utils import setup_logger

logger = setup_logger(__name__)

class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        try:
            words = text.split()
            chunks = []
            start = 0
            
            while start < len(words):
                end = start + self.chunk_size
                chunk = ' '.join(words[start:end])
                chunks.append(chunk)
                start = end - self.overlap

            logger.info(f"Text chunked into {len(chunks)} parts")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise

    def set_chunk_size(self, size: int) -> None:
        self.chunk_size = size
        logger.info(f"Chunk size set to {size}")

    def set_overlap(self, overlap: int) -> None:
        self.overlap = overlap
        logger.info(f"Overlap set to {overlap}")

text_chunker = TextChunker()
```


## Archivo: config.py
### Ruta Relativa: ../src\core\config.py

```python
import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class Settings:
    MONGODB_URI: str = os.getenv("MONGODB_URI")
    MONGODB_USER: str = os.getenv("MONGODB_USER")
    MONGODB_PASS: str = os.getenv("MONGODB_PASS")
    MONGODB_DB: str = os.getenv("MONGODB_DB", "cashabotorllm")
    
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Configuración para futuros modelos
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    DEBUGG: str = os.getenv("ENV") == "development"

    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "ollama": {
            "base_url": OLLAMA_BASE_URL,
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": OPENAI_API_KEY,
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com",
            "api_k,ey": ANTHROPIC_API_KEY,
        },
        "huggingface": {
            "base_url": "https://api.huggingface.co",
            "api_key": HUGGINGFACE_API_KEY,
        },
        "groq": {
            "base_url": "https://api.groq.com",
            "api_key": GROQ_API_KEY
        },
    }
    
    DEFAULT_MODELS: Dict[str, str] = {
        "gpt-4o-mini": "openai",
        "gpt-4o-coder": "grok",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "huggingface",
        "claude-3-5-sonnet-20240620": "anthropic",
        "llama-3.1-70b-versatile": "groq",  # Asegúrate de que esté aquí
        "llama-3.1-405b-reasoning": "groq",  # Asegúrate de que esté aquí también
        "llama3-groq-70b-8192-tool-use-preview": "groq"
    }

    @property
    def DATABASE_URL(self):
        return f"mongodb://{self.MONGODB_USER}:{self.MONGODB_PASS}@{self.MONGODB_URI.split('://')[1]}"

settings = Settings()
```


## Archivo: database.py
### Ruta Relativa: ../src\core\storage\database.py

```python
from pymongo import MongoClient
from bson import ObjectId
from src.core.config import settings
from src.core.utils import setup_logger
from typing import Dict, Any, List
import traceback

logger = setup_logger(__name__)

class Database:
    def __init__(self):
        try:
            logger.info(f"Connecting to database at {settings.DATABASE_URL}")
            self.client = MongoClient(settings.DATABASE_URL)
            self.db = self.client[settings.MONGODB_DB]
            self._ensure_database_exists()
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _ensure_database_exists(self):
        if settings.MONGODB_DB not in self.client.list_database_names():
            logger.info(f"Creating database: {settings.MONGODB_DB}")
            self.db.create_collection("dummy")
            self.db.drop_collection("dummy")
        logger.info(f"Using database: {settings.MONGODB_DB}")

    def _serialize_object_id(self, obj):
        if isinstance(obj, dict):
            return {key: self._serialize_object_id(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_object_id(item) for item in obj]
        elif isinstance(obj, ObjectId):
            return str(obj)
        return obj

    def store_embedding(self, embedding: List[float], metadata: Dict[str, Any]) -> str:
        try:
            collection = self.db.embeddings
            logger.info(f"Storing embedding in collection: {collection.name}")
            result = collection.insert_one({"embedding": embedding, "metadata": metadata})
            embedding_id = str(result.inserted_id)
            logger.info(f"Stored embedding with id: {embedding_id}")
            return embedding_id
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def search_similar_embeddings(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            collection = self.db.embeddings
            logger.info(f"Searching for similar embeddings in collection: {collection.name}")
            logger.info(f"Query embedding length: {len(query_embedding)}")
            
            pipeline = [
                {
                    "$addFields": {
                        "dot_product": {
                            "$reduce": {
                                "input": {"$zip": {"inputs": ["$embedding", query_embedding]}},
                                "initialValue": 0,
                                "in": {"$add": ["$$value", {"$multiply": [{"$arrayElemAt": ["$$this", 0]}, {"$arrayElemAt": ["$$this", 1]}]}]}
                            }
                        },
                        "magnitude_a": {
                            "$sqrt": {
                                "$reduce": {
                                    "input": "$embedding",
                                    "initialValue": 0,
                                    "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                                }
                            }
                        },
                        "magnitude_b": {
                            "$sqrt": {
                                "$reduce": {
                                    "input": query_embedding,
                                    "initialValue": 0,
                                    "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                                }
                            }
                        }
                    }
                },
                {
                    "$addFields": {
                        "cosine_similarity": {
                            "$divide": ["$dot_product", {"$multiply": ["$magnitude_a", "$magnitude_b"]}]
                        }
                    }
                },
                {"$sort": {"cosine_similarity": -1}},
                {"$limit": top_k},
                {
                    "$project": {
                        "_id": 1,
                        "metadata": 1,
                        "cosine_similarity": 1
                    }
                }
            ]
            
            logger.info("Executing aggregation pipeline...")
            results = list(collection.aggregate(pipeline))
            logger.info(f"Found {len(results)} similar embeddings")
            
            # Ensure the correct format of the results and handle potential missing fields
            formatted_results = []
            for result in results:
                formatted_result = {
                    "_id": str(result["_id"]),
                    "metadata": result.get("metadata", {}),
                    "cosine_similarity": result.get("cosine_similarity", 0.0)
                }
                formatted_results.append(formatted_result)
                logger.info(f"Formatted result: {formatted_result}")
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    # Nuevas funciones para soporte RAG

    def store_document(self, content: str, metadata: Dict[str, Any]) -> str:
        try:
            collection = self.db.documents
            logger.info(f"Storing document in collection: {collection.name}")
            result = collection.insert_one({"content": content, "metadata": metadata})
            document_id = str(result.inserted_id)
            logger.info(f"Stored document with id: {document_id}")
            return document_id
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_document_by_id(self, document_id: str) -> Dict[str, Any]:
        try:
            collection = self.db.documents
            document = collection.find_one({"_id": ObjectId(document_id)})
            if document:
                return self._serialize_object_id(document)
            else:
                logger.warning(f"Document with id {document_id} not found")
                return None
        except Exception as e:
            logger.error(f"Error retrieving document: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def update_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        try:
            collection = self.db.documents
            result = collection.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": {"content": content, "metadata": metadata}}
            )
            if result.modified_count > 0:
                logger.info(f"Updated document with id: {document_id}")
                return True
            else:
                logger.warning(f"Document with id {document_id} not found or not modified")
                return False
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            logger.error(traceback.format_exc())
            raise

db = Database()
```


## Archivo: models.py
### Ruta Relativa: ../src\core\storage\models.py

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime

class Embedding(BaseModel):
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    model_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class User(BaseModel):
    user_id: str
    username: str
    email: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
```


## Archivo: utils.py
### Ruta Relativa: ../src\core\utils.py

```python
import logging

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def sanitize_input(input_string: str) -> str:
    # Implementa la lógica de sanitización aquí
    return input_string.strip()
```


## Archivo: Class.py
### Ruta Relativa: ../src\entity\Class.py

```python
from typing import Any, Dict, Optional, List, Union
from pydantic import Field
from pydantic import BaseModel as PBaseModel, Field

class Message(PBaseModel):
    role: str
    content: str

class ChatRequest(PBaseModel):
    # Parámetros obligatorios
    model_name: str
    messages: List[Message]

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
    num_predict: Optional[int] = None
    repeat_last_n: Optional[int] = None
    tfs_z: Optional[float] = None
    num_ctx: Optional[int] = None
    num_batch: Optional[int] = None
    num_gqa: Optional[int] = None
    num_gpu: Optional[int] = None
    main_gpu: Optional[int] = None
    low_vram: Optional[bool] = None
    f16_kv: Optional[bool] = None
    logits_all: Optional[bool] = None
    vocab_only: Optional[bool] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    embedding_only: Optional[bool] = None
    rope_frequency_base: Optional[float] = None
    rope_frequency_scale: Optional[float] = None
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
```


## Archivo: anthropic.py
### Ruta Relativa: ../src\models\client\anthropic.py

```python
from typing import Any, Dict, Optional, List
from anthropic import Anthropic
from ...contract.IClient import IClient
from ...core.utils import setup_logger
from ...core.config import settings
import json

logger = setup_logger(__name__)

class AnthropicModel(IClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def load(self) -> None:
        # No es necesario cargar explícitamente para Anthropic
        pass

    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> Dict[str, Any]:
        try:
            filtered_kwargs = self._filter_kwargs(kwargs)
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **filtered_kwargs
            )
            return self._format_response(response)
        except Exception as e:
            logger.error(f"Error generating with Anthropic model {self.model_name}: {str(e)}")
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
            
            filtered_kwargs = self._filter_kwargs(**kwargs)
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or 1024,
                temperature=temperature,
                system=system_message if system_message else None,
                messages=filtered_messages,
                **filtered_kwargs
            )
            return self._format_response(response)
        except Exception as e:
            logger.error(f"Error generating chat with Anthropic model {self.model_name}: {str(e)}")
            raise

    def _format_response(self, response) -> Dict[str, Any]:
        content = response.content
        # Si content es una lista, tomamos el primer elemento
        if isinstance(content, list) and len(content) > 0:
            content = content[0].text if hasattr(content[0], 'text') else str(content[0])
        else:
            content = str(content)

        return {
            "message": content,
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
        return {"name": self.model_name, "type": "anthropic"}

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
                model=self.model_name,
                messages=[
                    {"role": "system", "content": task_description},
                    {"role": "user", "content": user_input}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error in process_auto_agent with Anthropic model {self.model_name}: {str(e)}")
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
```


## Archivo: groq.py
### Ruta Relativa: ../src\models\client\groq.py

```python
from typing import Any, Dict, Optional, List
from ...contract.IClient import IClient
from ...core.config import settings
from ...core.utils import setup_logger
from groq import Groq

logger = setup_logger(__name__)

class GroqModel(IClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def load(self) -> None:
        # Groq models don't need to be explicitly loaded
        pass

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> Optional[object]:
        if (settings.DEBUGG):
            for key, value in kwargs.items():
                print(f"{key}: {value}")
        try:
            filter_kwargs = self._filter_kwargs(**kwargs)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **filter_kwargs
            )
            
            response_dict = {
                "message": {"content": response.choices[0].message.content},
                "done_reason": response.choices[0].finish_reason,
                "done": True,
                "total_duration": response.usage.total_tokens,
                "prompt_eval_count": response.usage.prompt_tokens,
                "eval_count": response.usage.completion_tokens,
            }
            
            return response_dict
        except Exception as e:
            raise e
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating text with Groq model {self.model_name}: {str(e)}")
            raise
        
    def _filter_kwargs(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        accepted_params = [
            'max_tokens', 'temperature', 'top_p', 'stream',
            'stop', 'presence_penalty', 'frequency_penalty'
        ]
        return {k: v for k, v in kwargs.items() if k in accepted_params}

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "groq"}

    def generate_embedding(self, text: str) -> str:
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding with Groq model {self.model_name}: {str(e)}")
            raise

    def create_chunks(self, content: str, content_type: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def process_auto_agent(self, user_input: str, task_description: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def get_models(self) -> List[Dict[str, str]]:
        try:
            # Note: Groq might not have an API endpoint to list models
            # This is a placeholder implementation
            return [{"id": self.model_name, "object": "model"}]
        except Exception as e:
            logger.error(f"Error getting models from Groq: {str(e)}")
            raise

    def generate_prompt(self, prompt: str) -> str:
        # Placeholder implementation as details are needed
        return prompt

    def generate_prompts(self, messages: List[Dict[str, str]]) -> str:
        # Placeholder implementation as details are needed
        return " ".join([message["content"] for message in messages])
```


## Archivo: huggingface.py
### Ruta Relativa: ../src\models\client\huggingface.py

```python
from typing import Any, Dict, Optional, List
from ...contract.IClient import IClient
from ...core.config import settings
from ...core.utils import setup_logger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

logger = setup_logger(__name__)

class HuggingFaceModel(IClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.hf_token = settings.HUGGINGFACE_API_KEY  # Asume que has añadido este token a tu configuración

    def load(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=self.hf_token)
            logger.info(f"Successfully loaded model and tokenizer for {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            raise

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> Optional[object]:
        if not self.model or not self.tokenizer:
            self.load()

        try:
            # Concatenate messages into a single string
            prompt = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            filter_kwargs = self._filter_kwargs(kwargs)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **filter_kwargs
            )
            
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            response_dict = {
                "message": {"content": response_text},
                "done_reason": "stop",
                "done": True,
                "total_duration": len(outputs[0]),
                "prompt_eval_count": len(inputs.input_ids[0]),
                "eval_count": len(outputs[0]) - len(inputs.input_ids[0]),
            }
            
            return response_dict
        except Exception as e:
            logger.error(f"Error generating chat with HuggingFace model {self.model_name}: {str(e)}")
            raise
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        if not self.model or not self.tokenizer:
            self.load()

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating text with HuggingFace model {self.model_name}: {str(e)}")
            raise
        
    def _filter_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        accepted_params = [
            'max_length', 'min_length', 'do_sample', 'early_stopping', 'num_beams',
            'temperature', 'top_k', 'top_p', 'repetition_penalty', 'bad_words_ids',
            'bos_token_id', 'pad_token_id', 'eos_token_id', 'length_penalty',
            'no_repeat_ngram_size', 'encoder_no_repeat_ngram_size', 'num_return_sequences',
            'max_time', 'max_new_tokens', 'decoder_start_token_id', 'use_cache',
            'num_beam_groups', 'diversity_penalty', 'prefix_allowed_tokens_fn',
            'output_attentions', 'output_hidden_states', 'output_scores', 'return_dict_in_generate',
            'forced_bos_token_id', 'forced_eos_token_id', 'remove_invalid_values',
            'exponential_decay_length_penalty', 'suppress_tokens', 'begin_suppress_tokens',
            'forced_decoder_ids', 'sequence_bias', 'guidance_scale', 'low_memory'
        ]
        return {k: v for k, v in kwargs.items() if k in accepted_params}

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "huggingface"}

    def generate_embedding(self, text: str) -> List[float]:
        if not self.model or not self.tokenizer:
            self.load()

        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        except Exception as e:
            logger.error(f"Error generating embedding with HuggingFace model {self.model_name}: {str(e)}")
            raise

    # Los demás métodos permanecen sin cambios

    def create_chunks(self, content: str, content_type: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def process_auto_agent(self, user_input: str, task_description: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def get_models(self) -> List[Dict[str, str]]:
        # Hugging Face has a vast number of models, so we'll just return the current model
        return [{"id": self.model_name, "object": "model"}]

    def generate_prompt(self, prompt: str) -> str:
        # Placeholder implementation as details are needed
        return prompt

    def generate_prompts(self, messages: List[Dict[str, str]]) -> str:
        # Placeholder implementation as details are needed
        return " ".join([message["content"] for message in messages])
```


## Archivo: ollama.py
### Ruta Relativa: ../src\models\client\ollama.py

```python
import requests
import json
from typing import Any, Dict, Optional, List
from ...contract.IClient import IClient
from ...core.config import settings
from ...core.utils import setup_logger

logger = setup_logger(__name__)

class OllamaModel(IClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = settings.OLLAMA_BASE_URL

    def load(self) -> None:
        if not self._is_model_available():
            raise ValueError(f"Model {self.model_name} is not available in Ollama")

    def _is_model_available(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [model["name"] for model in models]
                return any(self.model_name in model for model in available_models)
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama model availability: {str(e)}")
            return False

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> str:
        try:
            formatted_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            formatted_messages += "\nassistant:"
            
            payload = {
                "model": self.model_name,
                "prompt": formatted_messages
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if temperature is not None:
                payload["temperature"] = temperature
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            if response.status_code == 200:
                response_text = response.text
                response_lines = response_text.strip().split('\n')
                full_response = ""
                for line in response_lines:
                    try:
                        response_json = json.loads(line)
                        full_response += response_json.get("response", "")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line as JSON: {line}")
                return full_response.strip()
            else:
                raise Exception(f"Ollama API error: {response.text}")
        except Exception as e:
            logger.error(f"Error generating chat with Ollama model {self.model_name}: {str(e)}")
            raise
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
            payload["temperature"] = temperature
            
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            if response.status_code == 200:
                response_text = response.text
                response_lines = response_text.strip().split('\n')
                full_response = ""
                for line in response_lines:
                    try:
                        response_json = json.loads(line)
                        full_response += response_json.get("response", "")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line as JSON: {line}")
                return full_response.strip()
            else:
                raise Exception(f"Ollama API error: {response.text}")
        except Exception as e:
            logger.error(f"Error generating with Ollama model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "ollama"}
```


## Archivo: ollamaSDK.py
### Ruta Relativa: ../src\models\client\ollamaSDK.py

```python
import json
import ollama
from typing import Any, Dict, Optional, List
from ...contract.IClient import IClient
from ...core.config import settings
from ...core.utils import setup_logger

logger = setup_logger(__name__)

class OllamaModel(IClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.base_url = settings.OLLAMA_BASE_URL

    def load(self) -> None:
        if not self._is_model_available():
            logger.info(f"Model {self.model_name} is not available. Attempting to load it.")
            if not self._load_model():
                raise ValueError(f"Model {self.model_name} could not be loaded in Ollama")

    def _is_model_available(self) -> bool:
        try:
            response = ollama.get_models()
            available_models = [model["name"] for model in response]
            return self.model_name in available_models
        except AttributeError:
            logger.error("The method 'get_models' does not exist in the ollama module")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama model availability: {str(e)}")
            return False

    def _load_model(self) -> bool:
        try:
            response = ollama.load_model(self.model_name)
            if response.get('status') == 'success':
                return True
            else:
                logger.error(f"Failed to load model {self.model_name}: {response.get('message', 'Unknown error')}")
                return False
        except AttributeError:
            logger.error("The method 'load_model' does not exist in the ollama module")
            return False
        except Exception as e:
            logger.error(f"Error loading Ollama model {self.model_name}: {str(e)}")
            return False

    @staticmethod
    def get_available_models() -> List[str]:
        try:
            response = ollama.get_models()
            available_models = [model["name"] for model in response]
            return available_models
        except AttributeError:
            logger.error("The method 'get_models' does not exist in the ollama module")
            return []
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")
            return []

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> str:
        try:
            response = ollama.chat(model=self.model_name, messages=messages)
            if 'message' in response:
                response_text = response['message']
                full_response = ""
                for line in response_text.split('\n'):
                    try:
                        response_json = json.loads(line)
                        full_response += response_json.get("response", "")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line as JSON: {line}")
                return full_response.strip()
            else:
                raise Exception(f"Ollama API error: {response}")
        except Exception as e:
            logger.error(f"Error generating chat with Ollama model {self.model_name}: {str(e)}")
            raise
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        try:
            response = ollama.generate(model=self.model_name, prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            if 'message' in response:
                response_text = response['message']
                full_response = ""
                for line in response_text.split('\n'):
                    try:
                        response_json = json.loads(line)
                        full_response += response_json.get("response", "")
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line as JSON: {line}")
                return full_response.strip()
            else:
                raise Exception(f"Ollama API error: {response}")
        except Exception as e:
            logger.error(f"Error generating with Ollama model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "ollama"}
```


## Archivo: ollamaSDKV2.py
### Ruta Relativa: ../src\models\client\ollamaSDKV2.py

```python
import json
import ollama
from typing import Any, Dict, Optional, List
from ...contract.IClient import IClient
from ...core.config import settings
from ...core.utils import setup_logger

logger = setup_logger(__name__)

class OllamaModel(IClient):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def load(self) -> None:
        # Assuming Ollama models don't need explicit loading
        pass

    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response['text'].strip()
        except Exception as e:
            logger.error(f"Error generating text with Ollama model {self.model_name}: {str(e)}")
            raise

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> Optional[object]:
        try:
            filter_kwargs = self._filter_kwargs(**kwargs)
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                **filter_kwargs
            )
            return response
        except Exception as e:
            logger.error(f"Error generating chat with Ollama model {self.model_name}: {str(e)}")
            raise
        
    def _filter_kwargs(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        accepted_params = [
            'num_predict', 'top_k', 'temperature', 'repeat_penalty',
            'repeat_last_n', 'seed', 'stop', 'num_ctx', 'num_batch',
            'num_gpu', 'main_gpu', 'low_vram', 'f16_kv', 'logits_all',
            'vocab_only', 'use_mmap'
        ]
        return {k: v for k, v in kwargs.items() if k in accepted_params and v is not None}

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "ollama"}

    def generate_embedding(self, text: str) -> str:
        try:
            response = ollama.embedding(
                model=self.model_name,
                input=text
            )
            return response['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding with Ollama model {self.model_name}: {str(e)}")
            raise

    def create_chunks(self, content: str, content_type: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def process_auto_agent(self, user_input: str, task_description: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def get_models(self) -> List[Dict[str, str]]:
        try:
            response = ollama.list_models()
            return [{"id": model['id'], "object": model['object']} for model in response['models']]
        except Exception as e:
            logger.error(f"Error getting models from Ollama: {str(e)}")
            raise

    def generate_prompt(self, prompt: str) -> str:
        # Placeholder implementation as details are needed
        return prompt

    def generate_prompts(self, messages: List[Dict[str, str]]) -> str:
        # Placeholder implementation as details are needed
        return " ".join([message["content"] for message in messages])
```


## Archivo: openai.py
### Ruta Relativa: ../src\models\client\openai.py

```python
from openai import OpenAI
from typing import Any, Dict, Optional, List
from ...contract.IClient import IClient
from ...core.config import settings
from ...core.utils import setup_logger

logger = setup_logger(__name__)

class OpenAIModel(IClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def load(self) -> None:
        # OpenAI models don't need to be explicitly loaded
        pass

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> Optional[object]:
        for key, value in kwargs.items():
            print(f"{key}: {value}")
        try:
            filter_kwargs = self._filter_kwargs(**kwargs)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **filter_kwargs
            )
            
            # Convert the response to a dictionary
            response_dict = {
                "message": {"content": response.choices[0].message.content},
                "done_reason": response.choices[0].finish_reason,
                "done": True,
                "total_duration": response.usage.total_tokens,  # This is not exactly duration, but a close approximation
                "prompt_eval_count": response.usage.prompt_tokens,
                "eval_count": response.usage.completion_tokens,
            }
            
            return response_dict
        except Exception:
            raise
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating text with OpenAI model {self.model_name}: {str(e)}")
            raise
        
    def _filter_kwargs(self, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        accepted_params = [
            'model', 'temperature', 'top_p', 'n', 'stream', 'stop', 'max_tokens',
            'presence_penalty', 'frequency_penalty', 'logit_bias', 'user',
            'response_format', 'seed', 'tools', 'tool_choice'
        ]
        return {k: v for k, v in kwargs.items() if k in accepted_params and v is not None}

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "openai"}

    def generate_embedding(self, text: str) -> str:
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding with OpenAI model {self.model_name}: {str(e)}")
            raise

    def create_chunks(self, content: str, content_type: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def process_auto_agent(self, user_input: str, task_description: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def get_models(self) -> List[Dict[str, str]]:
        try:
            response = self.client.models.list()
            return [{"id": model.id, "object": model.object} for model in response.data]
        except Exception as e:
            logger.error(f"Error getting models from OpenAI: {str(e)}")
            raise

    def generate_prompt(self, prompt: str) -> str:
        # Placeholder implementation as details are needed
        return prompt

    def generate_prompts(self, messages: List[Dict[str, str]]) -> str:
        # Placeholder implementation as details are needed
        return " ".join([message["content"] for message in messages])
```


## Archivo: embeddings.py
### Ruta Relativa: ../src\models\embeddings.py

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from ..core.utils import setup_logger

logger = setup_logger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def compare_embeddings(self, embedding1: List[float], embedding2: List[float]) -> float:
        try:
            return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        except Exception as e:
            logger.error(f"Error comparing embeddings: {str(e)}")
            raise

embedding_generator = EmbeddingGenerator()
```


## Archivo: model_manager.py
### Ruta Relativa: ../src\models\model_manager.py

```python
import requests
from typing import Dict, Any, List, Optional
from ..contract.IClient import IClient
from .client.ollamaSDKV2 import OllamaModel
from .client.huggingface import HuggingFaceModel
from .client.openai import OpenAIModel
from .client.anthropic import AnthropicModel
from .client.groq import GroqModel
from ..core.utils import setup_logger
from ..core.config import settings

logger = setup_logger(__name__)

class ModelManager:
    def __init__(self):
        self.models: Dict[str, IClient] = {}
        self.default_models = settings.DEFAULT_MODELS
        self._load_ollama_models()
        logger.info("Initializing ModelManager")

    def _load_ollama_models(self) -> None:
        try:
            response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                ollama_model_names = [model["name"] for model in models]
                for model_name in ollama_model_names:
                    self.default_models[model_name] = "ollama"
                    self.default_models[f"{model_name}:latest"] = "ollama"
                logger.info(f"Ollama models loaded: {ollama_model_names}")
            else:
                logger.error(f"Failed to get Ollama models. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")

    def _load_model(self, model_name: str, model_type: str) -> None:
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            return

        model_classes = {
            "ollama": OllamaModel,
            "huggingface": HuggingFaceModel,
            "openai": OpenAIModel,
            "anthropic": AnthropicModel,
            "groq": GroqModel
        }

        if model_type not in model_classes:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        try:
            logger.info(f"Creating instance of {model_type} model: {model_name}")
            model = model_classes[model_type](model_name)
            logger.info(f"Loading model: {model_name}")
            model.load()
            self.models[model_name] = model
            logger.info(f"Model {model_name} of type {model_type} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise

    def load_model(self, model_name: str, model_type: str) -> None:
        self._load_model(model_name, model_type)

    def get_model(self, model_name: str, model_type: Optional[str] = None) -> Optional[IClient]:
        if model_name not in self.models:
            if model_type is None:
                model_type = self.default_models.get(model_name)
                if model_type is None:
                    logger.error(f"Model type for {model_name} not provided and not found in default models")
                    return None
            try:
                self.load_model(model_name, model_type)
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                return None
        return self.models.get(model_name)

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        return [model.get_info() for model in self.models.values()]

    def _generate_response(self, model_name: str, input_data: Any, max_tokens: Optional[int], temperature: float, model_type: Optional[str], method: str, **kwargs) -> Optional[Any]:
        try:
            model = self.get_model(model_name, model_type)
            if model is None:
                logger.error(f"Model {model_name} not loaded")
                return f"Model {model_name} not loaded"

            if method == "generate":
                return model.generate(input_data, max_tokens, temperature, **kwargs)
            elif method == "chat":
                return model.generate_chat(input_data, max_tokens, temperature, **kwargs)
            else:
                logger.error(f"Unsupported method: {method}")
                return f"Unsupported method: {method}"
        except Exception as e:
            raise e

    def generate(self, model_name: str, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7, model_type: Optional[str] = None, **kwargs) -> Any:
        return self._generate_response(model_name, prompt, max_tokens, temperature, model_type, method="generate", **kwargs)

    def generate_chat(self, model_name: str, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, model_type: Optional[str] = None, **kwargs) -> Optional[object]:
        messages_dict = [msg.dict() for msg in messages]
        try:
            resp = self._generate_response(model_name, messages_dict, max_tokens, temperature, model_type, method="chat", **kwargs)
            return resp
        except Exception as e:
            raise e

model_manager = ModelManager()
```


## Archivo: anthropic_prompt.py
### Ruta Relativa: ../src\models\prompts\anthropic_prompt.py

```python
from typing import List, Dict, Any
from .base_prompt import BasePromptHandler

class AnthropicPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        system_message = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
        chat_messages = [msg for msg in messages if msg['role'] != 'system']
        return {
            "system": system_message,
            "messages": chat_messages
        }

```


## Archivo: base_prompt.py
### Ruta Relativa: ../src\models\prompts\base_prompt.py

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BasePromptHandler(ABC):
    @abstractmethod
    def format_prompt(self, messages: List[Dict[str, str]]) -> Any:
        pass

```


## Archivo: grok_prompt.py
### Ruta Relativa: ../src\models\prompts\grok_prompt.py

```python
from typing import List, Dict
from .base_prompt import BasePromptHandler

class GrokPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

```


## Archivo: huggingface_prompt.py
### Ruta Relativa: ../src\models\prompts\huggingface_prompt.py

```python
from typing import List, Dict
from .base_prompt import BasePromptHandler

class HuggingFacePromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        return " ".join([msg['content'] for msg in messages])

```


## Archivo: ollama_prompt.py
### Ruta Relativa: ../src\models\prompts\ollama_prompt.py

```python
from typing import List, Dict, Any
from .base_prompt import BasePromptHandler

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

```


## Archivo: openai_prompt.py
### Ruta Relativa: ../src\models\prompts\openai_prompt.py

```python
from typing import List, Dict, Any

class OpenAIPromptHandler:
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
```


## Archivo: prompt_factory.py
### Ruta Relativa: ../src\models\prompts\prompt_factory.py

```python
from .openai_prompt import OpenAIPromptHandler
from .ollama_prompt import OllamaPromptHandler
from .anthropic_prompt import AnthropicPromptHandler
from .grok_prompt import GrokPromptHandler
from .huggingface_prompt import HuggingFacePromptHandler

class PromptHandlerFactory:
    @staticmethod
    def get_handler(model_type: str):
        handlers = {
            "openai": OpenAIPromptHandler(),
            "ollama": OllamaPromptHandler(),
            "anthropic": AnthropicPromptHandler(),
            "grok": GrokPromptHandler(),
            "huggingface": HuggingFacePromptHandler()
        }
        return handlers.get(model_type.lower())

```


## Archivo: prompt_handler.py
### Ruta Relativa: ../src\models\prompts\prompt_handler.py

```python
from pydantic import BaseModel
from ..model_manager import model_manager
from ...core.utils import setup_logger
from .prompt_factory import PromptHandlerFactory
from typing import List, Optional

logger = setup_logger(__name__)

class Message(BaseModel):
    role: str
    content: str

class PromptHandler:
    def __init__(self):
        self.model_manager = model_manager

    def process_prompt(self, model_name: str, prompt_type: str, messages: List[Message], max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        try:
            model = self.model_manager.get_model(model_name)
            model_type = model.get_info()['type']
            handler = PromptHandlerFactory.get_handler(model_type)
            
            formatted_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
            
            if model_type == "openai":
                # Para modelos de OpenAI, pasamos los mensajes directamente
                return model.generate(formatted_messages, max_tokens=max_tokens, temperature=temperature)
            else:
                # Para otros tipos de modelos, usamos el handler para formatear el prompt
                formatted_prompt = handler.format_prompt(formatted_messages)
                return model.generate(formatted_prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}")
            raise

prompt_handler = PromptHandler()
```


## Archivo: downloadHFModel.py
### Ruta Relativa: ../src\tools\downloadHFModel.py

```python
import requests
import os

# Autenticarte en Hugging Face
huggingface_token = "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"  # Reemplaza con tu token de Hugging Face
headers = {"Authorization": f"Bearer {huggingface_token}"}

# Lista de archivos a descargar
files_to_download = [
    "model.safetensors.index.json",
    "modeling_intern_vit.py",
    "modeling_internlm2.py",
    "modeling_internvl_chat.py",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenization_internlm2.py",
    "tokenization_internlm2_fast.py",
    "tokenizer.model",
    "tokenizer_config.json",
]

# Directorio base de Hugging Face
base_url = "https://huggingface.co/OpenGVLab/InternVL2-26B/resolve/main/"

# Directorio donde se guardarán los archivos descargados
download_directory = "./InternVL2-26B/"
os.makedirs(download_directory, exist_ok=True)

# Función para descargar un archivo
def download_file(file_url, destination):
    response = requests.get(file_url, headers=headers, stream=True)
    if response.status_code == 200:
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {destination}")
    else:
        print(f"Failed to download {file_url}")

# Descargar cada archivo en la lista
for file_name in files_to_download:
    file_url = base_url + file_name
    destination = os.path.join(download_directory, file_name)
    download_file(file_url, destination)

```


## Archivo: index.py
### Ruta Relativa: ../src\tools\langchain\index.py

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Obtener las variables de entorno
api_key = os.getenv("OPENAI_API_KEY")
endpoint = os.getenv("LANGCHAIN_ENDPOINT")
tracing = os.getenv("LANGCHAIN_TRACING_V2")
project = os.getenv("LANGCHAIN_PROJECT")

# Verificar que las variables de entorno se están cargando correctamente
print(f"API Key: {api_key}")
print(f"Endpoint: {endpoint}")
print(f"Tracing: {tracing}")
print(f"Project: {project}")

# Inicializar el modelo de lenguaje con la clave API
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

# Invocar el modelo con un prompt
try:
    response = llm.invoke("Hello, world!")
    print(response)
except Exception as e:
    print(f"Error: {e}")

```


## Archivo: ollama.py
### Ruta Relativa: ../src\tools\langchain\ollama.py

```python
from langchain_community.llms import Ollama
ollama = Ollama(
    base_url='http://localhost:11434',
    model="mistral",
    temperature=0,
)
print(ollama.invoke("why is the sky blue"))
```


## Archivo: ollamaStream.py
### Ruta Relativa: ../src\tools\ollamaStream.py

```python
import asyncio
from ollama import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  response = await AsyncClient().chat(model='llama3.1', messages=[message])

asyncio.run(chat())
```


## Archivo: test-any.py
### Ruta Relativa: ../src\tools\test-any.py

```python
def int_to_string(n):
    return str(n)

# Pide un entero al usuario
user_input = input("Enter an integer: ")

try:
    # Convierte el input en un entero
    n = int(user_input)
    # Convierte el entero en una cadena y muestra el resultado
    print(f"The integer {n} converted to a string is: {int_to_string(n)}")
except ValueError:
    print("Invalid input. Please enter an integer.")

```


## Archivo: test.py
### Ruta Relativa: ../src\tools\test.py

```python
from transformers import AutoModelForCausalLM
import torch

# Importar el tokenizador personalizado
from InternVL226B.tokenization_internlm2 import InternLM2Tokenizer
from InternVL226B.tokenization_internlm2_fast import InternLM2TokenizerFast

# Directorio donde se guardaron los archivos descargados
model_directory = "./InternVL226B/"

# Cargar el tokenizador y el modelo desde el directorio local
tokenizer = InternLM2Tokenizer.from_pretrained(model_directory)
# Alternativamente, si usas el tokenizador rápido
# tokenizer = InternLM2TokenizerFast.from_pretrained(model_directory)

model = AutoModelForCausalLM.from_pretrained(model_directory, trust_remote_code=True)

# Verificar la carga correcta
print("Modelo y tokenizador cargados correctamente.")

# Ejemplo de texto de entrada para el modelo
input_text = "Once upon a time"

# Tokenizar la entrada
inputs = tokenizer(input_text, return_tensors="pt")

# Generar texto con el modelo
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,  # Máxima longitud del texto generado
        num_return_sequences=1,  # Número de secuencias a generar
        no_repeat_ngram_size=2,  # Evita la repetición de n-gramas
        early_stopping=True
    )

# Decodificar y mostrar el texto generado
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Texto generado: {generated_text}")

```


## Archivo: testGrok.py
### Ruta Relativa: ../src\tools\testGrok.py

```python
import os
from groq import Groq

# Asegúrate de tener tu API key configurada en las variables de entorno
os.environ["GROQ_API_KEY"] = "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"

# Inicializar el cliente de Groq con tu API key
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Crear una solicitud de chat para el modelo llama3-groq-70b-8192-tool-use-preview
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Escribe una función en Python que tome una lista de números avansado y devuelva una nueva lista que contenga solo los números pares de la lista original. La función debe llamarse 'filtrar_numeros_pares'.",
        }
    ],
    # model="llama-3.1-70b-versatile",
    # model="llama-3.1-8b-instant",
    # model="llama3-groq-70b-8192-tool-use-preview",
    model="llama3.1:8b-instruct-fp16"
    temperature=0
)

# Imprimir el contenido de la respuesta
print(chat_completion.choices[0].message.content)

```


## Archivo: testLlama3.1Tools.py
### Ruta Relativa: ../src\tools\testLlama3.1Tools.py

```python
import ollama

response = ollama.chat(
    model='llama3.1',
    messages=[
        {"role": "system", "content": "Eres un asistente que ayuda a los usuarios con información precisa y detallada."},
        {"role": "user", "content": "Hola, ¿cómo estás?"},
        {"role": "assistant", "content": "Hola! Estoy aquí para ayudarte con cualquier pregunta que tengas."},
        {"role": "user", "content": "¿Puedes darme una recomendación de libros sobre inteligencia artificial?"}
    ],

		# provide a weather checking tool to the model
    tools=[{
      'type': 'function',
      'function': {
        'name': 'get_current_weather',
        'description': 'Get the current weather for a city',
        'parameters': {
          'type': 'object',
          'properties': {
            'city': {
              'type': 'string',
              'description': 'The name of the city',
            },
          }, 'required': ['city'],
        },
      },
    },
  ],
)

print(response['message']['tool_calls'])
```


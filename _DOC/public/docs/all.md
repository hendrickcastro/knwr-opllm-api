
## Archivo: autoagent.py
### Ruta Relativa: ../api\agents\autoagent.py

```python
from typing import Dict, Any
from models.model_manager import model_manager
from prompts.prompt_handler import prompt_handler
from core.utils import setup_logger

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


## Archivo: routes.py
### Ruta Relativa: ../api\api\routes.py

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
from models.model_manager import model_manager
from prompts.prompt_handler import prompt_handler
from chunks.chunk_handler import chunk_handler
from models.embeddings import embedding_generator
from storage.database import db
from agents.autoagent import auto_agent_factory
import logging
import traceback

logger = logging.getLogger(__name__)

router = APIRouter()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model_name: str
    messages: List[Message]

class ChatResponse(BaseModel):
    response: str

class EmbeddingRequest(BaseModel):
    text: str

    class Config:
        protected_namespaces = ()

class EmbeddingResponse(BaseModel):
    embedding: List[float]

class ChunkRequest(BaseModel):
    content: str
    content_type: str

    class Config:
        protected_namespaces = ()

class ChunkResponse(BaseModel):
    chunks: List[str]

class AutoAgentRequest(BaseModel):
    model_name: str = Field(..., alias='model_name')
    task_description: str
    user_input: str

    class Config:
        protected_namespaces = ()

class AutoAgentResponse(BaseModel):
    response: str

class GenerateRequest(BaseModel):
    model: str = Field(..., alias='model')
    prompt: str
    max_tokens: Optional[int] = None

    class Config:
        protected_namespaces = ()

class CompareEmbeddingsRequest(BaseModel):
    text1: str
    text2: str

class CompareEmbeddingsResponse(BaseModel):
    similarity: float

class StoreEmbeddingRequest(BaseModel):
    text: str
    metadata: Dict[str, Any]

class StoreEmbeddingResponse(BaseModel):
    embedding_id: str

class SearchSimilarEmbeddingsRequest(BaseModel):
    text: str
    top_k: int = 5

class SimilarEmbedding(BaseModel):
    id: str
    metadata: Dict[str, Any]
    cosine_similarity: float

class SearchSimilarEmbeddingsResponse(BaseModel):
    similar_embeddings: List[SimilarEmbedding]

class GenerateResponse(BaseModel):
    generated_text: str
    
class RAGRequest(BaseModel):
    query: str
    model_name: str
    top_k: int = 5

class RAGResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    
@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request for model: {request.model_name}")
        response = prompt_handler.process_prompt(request.model_name, "chat", messages=request.messages)
        return ChatResponse(response=response)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/embedding", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest):
    try:
        embedding = embedding_generator.generate_embedding(request.text)
        return EmbeddingResponse(embedding=embedding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chunk", response_model=ChunkResponse)
async def create_chunks(request: ChunkRequest):
    try:
        chunks = chunk_handler.process_chunks(request.content, request.content_type)
        return ChunkResponse(chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/autoagent", response_model=AutoAgentResponse)
async def process_auto_agent(request: AutoAgentRequest):
    try:
        agent = auto_agent_factory.create_agent(request.model_name, request.task_description)
        response = agent.process_input(request.user_input)
        return AutoAgentResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models", response_model=List[Dict[str, str]])
async def list_models():
    try:
        return model_manager.list_loaded_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        logger.info(f"Received generate request: {request.json()}")
        generated_text = model_manager.generate(
            request.model,  # Cambiado de request.model_name a request.model
            request.prompt,
            request.max_tokens
        )
        return GenerateResponse(generated_text=generated_text)
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/load_model")
async def load_model(model_name: str, model_type: str):
    try:
        model_manager.load_model(model_name, model_type)
        return {"message": f"Model {model_name} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/check_ollama")
async def check_ollama():
    try:
        return model_manager.check_ollama_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/check_ollama_models")
async def check_ollama_models():
    try:
        return model_manager.list_ollama_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare_embeddings", response_model=CompareEmbeddingsResponse)
async def compare_embeddings(request: CompareEmbeddingsRequest):
    try:
        embedding1 = embedding_generator.generate_embedding(request.text1)
        embedding2 = embedding_generator.generate_embedding(request.text2)
        similarity = embedding_generator.compare_embeddings(embedding1, embedding2)
        return CompareEmbeddingsResponse(similarity=similarity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/store_embedding", response_model=StoreEmbeddingResponse)
async def store_embedding(request: StoreEmbeddingRequest):
    try:
        embedding = embedding_generator.generate_embedding(request.text)
        embedding_id = db.store_embedding(embedding, request.metadata)
        return StoreEmbeddingResponse(embedding_id=embedding_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search_similar_embeddings", response_model=SearchSimilarEmbeddingsResponse)
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
    
@router.post("/rag", response_model=RAGResponse)
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


## Archivo: chunk_handler.py
### Ruta Relativa: ../api\chunks\chunk_handler.py

```python
from typing import List, Union
from .text_chunks import text_chunker
from .code_chunks import code_chunker
from core.utils import setup_logger

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
### Ruta Relativa: ../api\chunks\code_chunks.py

```python
import re
from typing import List
from core.utils import setup_logger

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
### Ruta Relativa: ../api\chunks\text_chunks.py

```python
from typing import List
from core.utils import setup_logger

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
### Ruta Relativa: ../api\core\config.py

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
    GROK_API_KEY: str = os.getenv("GROK_API_KEY")

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
        "grok": {
            "base_url": "https://api.grok.com",
            "api_key": GROK_API_KEY
        },
    }

    @property
    def DATABASE_URL(self):
        return f"mongodb://{self.MONGODB_USER}:{self.MONGODB_PASS}@{self.MONGODB_URI.split('://')[1]}"

settings = Settings()
```


## Archivo: utils.py
### Ruta Relativa: ../api\core\utils.py

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


## Archivo: main.py
### Ruta Relativa: ../api\main.py

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.routes import router
import uvicorn
from core.utils import setup_logger

logger = setup_logger(__name__)

app = FastAPI(debug=True)  # Enable debug mode

# Include the router without a prefix
app.include_router(router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"},
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```


## Archivo: anthropic_model.py
### Ruta Relativa: ../api\models\anthropic_model.py

```python
from typing import Any, Dict, Optional, List
from anthropic import Anthropic
from .base_model import BaseModel
from core.utils import setup_logger
from core.config import settings

logger = setup_logger(__name__)

class AnthropicModel(BaseModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    def load(self) -> None:
        # No es necesario cargar explícitamente
        pass

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or 1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating with Anthropic model {self.model_name}: {str(e)}")
            raise

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or 1024,
                messages=messages
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error generating chat with Anthropic model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "anthropic"}

# Asegúrate de que la clase esté siendo exportada
__all__ = ['AnthropicModel']
```


## Archivo: base_model.py
### Ruta Relativa: ../api\models\base_model.py

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List

class BaseModel(ABC):
    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        pass

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        # Default implementation for models that don't support chat natively
        prompt = self._create_chat_prompt(messages)
        return self.generate(prompt, max_tokens)

    def _create_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        for message in messages:
            role = message['role'].capitalize()
            content = message['content']
            prompt += f"{role}: {content}\n\n"
        prompt += "Assistant: "
        return prompt
```


## Archivo: embeddings.py
### Ruta Relativa: ../api\models\embeddings.py

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from core.utils import setup_logger

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


## Archivo: grok_model.py
### Ruta Relativa: ../api\models\grok_model.py

```python
from typing import Any, Dict, Optional
import requests
from .base_model import BaseModel
from core.utils import setup_logger
from core.config import settings

logger = setup_logger(__name__)

class GrokModel(BaseModel):
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = "https://api.grok.com"  # URL base de la API de Grok
        self.headers = {
            "Authorization": f"Bearer {settings.GROK_API_KEY}",
            "Content-Type": "application/json"
        }

    def load(self) -> None:
        # Aquí puedes agregar lógica para verificar si el modelo está disponible en Grok si es necesario
        pass

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "max_tokens": max_tokens or 100
            }
            
            logger.info(f"Sending payload to Grok API: {payload}")
            response = requests.post(f"{self.base_url}/generate", json=payload, headers=self.headers)
            logger.info(f"Grok API response status code: {response.status_code}")
            logger.info(f"Grok API response content: {response.text}")

            if response.status_code == 200:
                return response.json()["generated_text"]
            else:
                raise Exception(f"Grok API error: {response.text}")
        except Exception as e:
            logger.error(f"Error generating with Grok model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "grok"}

```


## Archivo: huggingface_model.py
### Ruta Relativa: ../api\models\huggingface_model.py

```python
from typing import Any, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseModel
from core.utils import setup_logger
from core.config import settings

logger = setup_logger(__name__)

class HuggingFaceModel(BaseModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = settings.HUGGINGFACE_API_KEY
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Error loading HuggingFace model {self.model_name}: {str(e)}")
            raise

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            max_length = inputs["input_ids"].shape[1] + (max_tokens or 100)
            outputs = self.model.generate(**inputs, max_length=max_length)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating with HuggingFace model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "huggingface"}
```


## Archivo: model_manager.py
### Ruta Relativa: ../api\models\model_manager.py

```python
from typing import Dict, Any, List, Optional
from .base_model import BaseModel
from .ollama_model import OllamaModel
from .huggingface_model import HuggingFaceModel
from .openai_model import OpenAIModel
from .anthropic_model import AnthropicModel
from .grok_model import GrokModel
from core.utils import setup_logger
import requests
from core.config import settings

logger = setup_logger(__name__)

class ModelManager:
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        logger.info("Initializing ModelManager")
        self._load_default_models()

    def _load_default_models(self):
        default_models = [
            ("gpt-4o-mini", "openai"),
            ("gpt-4o-coder", "grok"),
            ("microsoft/codebert-base", "huggingface"),
            ("claude-3-5-sonnet-20240620", "anthropic")
        ]
        logger.info(f"Attempting to load default models: {default_models}")
        for model_name, model_type in default_models:
            try:
                logger.info(f"Attempting to load model: {model_name} of type {model_type}")
                self.load_model(model_name, model_type)
            except Exception as e:
                logger.warning(f"Failed to load default model {model_name}: {str(e)}", exc_info=True)
        
        self._load_ollama_models()

    def _load_ollama_models(self):
        try:
            ollama_models = self._get_ollama_models()
            for model_name in ollama_models:
                try:
                    logger.info(f"Attempting to load Ollama model: {model_name}")
                    self.load_model(model_name, "ollama")
                except Exception as e:
                    logger.warning(f"Failed to load Ollama model {model_name}: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to load Ollama models: {str(e)}", exc_info=True)

    def _get_ollama_models(self) -> List[str]:
        try:
            response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                logger.error(f"Failed to get Ollama models. Status code: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")
            return []

    def load_model(self, model_name: str, model_type: str) -> None:
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            return

        model_classes = {
            "ollama": OllamaModel,
            "huggingface": HuggingFaceModel,
            "openai": OpenAIModel,
            "anthropic": AnthropicModel,
            "grok": GrokModel
        }

        if model_type not in model_classes:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info(f"Creating instance of {model_type} model: {model_name}")
        model = model_classes[model_type](model_name)
        logger.info(f"Loading model: {model_name}")
        model.load()
        self.models[model_name] = model
        logger.info(f"Model {model_name} of type {model_type} loaded successfully")

    def get_model(self, model_name: str) -> BaseModel:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        return self.models[model_name]

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        return [model.get_info() for model in self.models.values()]

    def generate(self, model_name: str, prompt: str, max_tokens: Optional[int] = None) -> str:
        model = self.get_model(model_name)
        return model.generate(prompt, max_tokens)

model_manager = ModelManager()
```


## Archivo: ollama_model.py
### Ruta Relativa: ../api\models\ollama_model.py

```python
import requests
import json
from typing import Any, Dict, Optional, List
from .base_model import BaseModel
from core.config import settings
from core.utils import setup_logger

logger = setup_logger(__name__)

class OllamaModel(BaseModel):
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

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
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

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        try:
            # Formato específico para Ollama
            formatted_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            formatted_messages += "\nassistant:"
            
            payload = {
                "model": self.model_name,
                "prompt": formatted_messages
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            
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

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "ollama"}
```


## Archivo: openai_model.py
### Ruta Relativa: ../api\models\openai_model.py

```python
from openai import OpenAI
from typing import Any, Dict, Optional
from .base_model import BaseModel
from core.config import settings
from core.utils import setup_logger

logger = setup_logger(__name__)

class OpenAIModel(BaseModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def load(self) -> None:
        # OpenAI models don't need to be explicitly loaded
        pass

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        try:
            # Determinar el tipo de modelo y el endpoint a utilizar
            if 'gpt-4o-mini' in self.model_name or 'gpt-3.5-turbo' in self.model_name or 'gpt-4' in self.model_name:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens or 100
                )
                return response.choices[0].message.content.strip()
            else:
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=max_tokens or 100
                )
                return response.choices[0].text.strip()
        except Exception as e:
            logger.error(f"Error generating with OpenAI model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "openai"}
```


## Archivo: anthropic_prompt_handler.py
### Ruta Relativa: ../api\prompts\anthropic_prompt_handler.py

```python
from typing import List, Dict, Any
from .base_prompt_handler import BasePromptHandler

class AnthropicPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        system_message = next((msg['content'] for msg in messages if msg['role'] == 'system'), None)
        chat_messages = [msg for msg in messages if msg['role'] != 'system']
        return {
            "system": system_message,
            "messages": chat_messages
        }

```


## Archivo: base_prompt_handler.py
### Ruta Relativa: ../api\prompts\base_prompt_handler.py

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BasePromptHandler(ABC):
    @abstractmethod
    def format_prompt(self, messages: List[Dict[str, str]]) -> Any:
        pass

```


## Archivo: prompt_handler.py
### Ruta Relativa: ../api\prompts\baul\prompt_handler.py

```python
from typing import List, Dict
from pydantic import BaseModel
from models.model_manager import model_manager
from core.utils import setup_logger

logger = setup_logger(__name__)

class Message(BaseModel):
    role: str
    content: str

class PromptHandler:
    def __init__(self):
        self.model_manager = model_manager

    def process_prompt(self, model_name: str, prompt_type: str, **kwargs) -> str:
        try:
            model = self.model_manager.get_model(model_name)
            
            if prompt_type == "chat":
                messages = kwargs.get('messages', [])
                # Convertir los objetos Message a diccionarios
                dict_messages = [{"role": msg.role, "content": msg.content} for msg in messages]
                return model.generate_chat(dict_messages)
            else:
                prompt = kwargs.get('input', '')
                return model.generate(prompt)
        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}")
            raise

prompt_handler = PromptHandler()
```


## Archivo: prompt_processor.py
### Ruta Relativa: ../api\prompts\baul\prompt_processor.py

```python
from typing import Dict, Any
from core.utils import setup_logger, sanitize_input

logger = setup_logger(__name__)

class PromptProcessor:
    def __init__(self):
        self.prompt_templates: Dict[str, str] = {
            "chat": "Human: {input}\nAI:",
            "completion": "{input}",
            "question_answering": "Context: {context}\nQuestion: {question}\nAnswer:",
        }

    def create_prompt(self, prompt_type: str, **kwargs) -> str:
        if prompt_type not in self.prompt_templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        try:
            sanitized_kwargs = {k: sanitize_input(v) for k, v in kwargs.items()}
            return self.prompt_templates[prompt_type].format(**sanitized_kwargs)
        except KeyError as e:
            logger.error(f"Missing required argument for prompt type {prompt_type}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error creating prompt: {str(e)}")
            raise

    def add_prompt_template(self, name: str, template: str) -> None:
        self.prompt_templates[name] = template
        logger.info(f"Added new prompt template: {name}")

prompt_processor = PromptProcessor()
```


## Archivo: grok_prompt_handler.py
### Ruta Relativa: ../api\prompts\grok_prompt_handler.py

```python
from typing import List, Dict
from .base_prompt_handler import BasePromptHandler

class GrokPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

```


## Archivo: huggingface_prompt_handler.py
### Ruta Relativa: ../api\prompts\huggingface_prompt_handler.py

```python
from typing import List, Dict
from .base_prompt_handler import BasePromptHandler

class HuggingFacePromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        return " ".join([msg['content'] for msg in messages])

```


## Archivo: ollama_prompt_handler.py
### Ruta Relativa: ../api\prompts\ollama_prompt_handler.py

```python
from typing import List, Dict
from .base_prompt_handler import BasePromptHandler

class OllamaPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> str:
        formatted_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return f"{formatted_messages}\nassistant:"

```


## Archivo: openai_prompt_handler.py
### Ruta Relativa: ../api\prompts\openai_prompt_handler.py

```python
from typing import List, Dict
from .base_prompt_handler import BasePromptHandler

class OpenAIPromptHandler(BasePromptHandler):
    def format_prompt(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return messages

```


## Archivo: prompt_handler.py
### Ruta Relativa: ../api\prompts\prompt_handler.py

```python
from pydantic import BaseModel
from models.model_manager import model_manager
from core.utils import setup_logger
from .prompt_handler_factory import PromptHandlerFactory

logger = setup_logger(__name__)

class Message(BaseModel):
    role: str
    content: str

class PromptHandler:
    def __init__(self):
        self.model_manager = model_manager

    def process_prompt(self, model_name: str, prompt_type: str, **kwargs) -> str:
        try:
            model = self.model_manager.get_model(model_name)
            model_type = model.get_info()['type']
            handler = PromptHandlerFactory.get_handler(model_type)
            
            if prompt_type == "chat":
                messages = kwargs.get('messages', [])
                formatted_messages = [msg.dict() for msg in messages]
                formatted_prompt = handler.format_prompt(formatted_messages)
            else:
                prompt = kwargs.get('input', '')
                formatted_prompt = handler.format_prompt([{"role": "user", "content": prompt}])
            
            return model.generate(formatted_prompt)
        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}")
            raise

prompt_handler = PromptHandler()

```


## Archivo: prompt_handler_factory.py
### Ruta Relativa: ../api\prompts\prompt_handler_factory.py

```python
from .openai_prompt_handler import OpenAIPromptHandler
from .ollama_prompt_handler import OllamaPromptHandler
from .anthropic_prompt_handler import AnthropicPromptHandler
from .grok_prompt_handler import GrokPromptHandler
from .huggingface_prompt_handler import HuggingFacePromptHandler

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


## Archivo: database.py
### Ruta Relativa: ../api\storage\database.py

```python
from pymongo import MongoClient
from bson import ObjectId
from core.config import settings
from core.utils import setup_logger
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
### Ruta Relativa: ../api\storage\models.py

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


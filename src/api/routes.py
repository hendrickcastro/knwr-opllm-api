from fastapi import APIRouter, HTTPException
from typing import List, Dict
from ..core.chunks.chunk_handler import chunk_handler
from ..core.agents.autoagent import auto_agent_factory
from ..models.model_manager import model_manager
from ..models.embeddings import embedding_generator
from ..core.storage.database import db
from ..entity.Class import ChatResponse, ChatRequest, EmbeddingRequest, EmbeddingResponse, ChunkRequest, ChunkResponse, AutoAgentRequest, AutoAgentResponse, GenerateRequest, GenerateResponse, CompareEmbeddingsRequest, CompareEmbeddingsResponse, StoreEmbeddingRequest, StoreEmbeddingResponse, SearchSimilarEmbeddingsRequest, SearchSimilarEmbeddingsResponse, RAGRequest, RAGResponse, SimilarEmbedding
import logging
import traceback

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
        kwargs = request.dict(exclude={"model_name", "messages"})
        
        response = model_manager.generate_chat(
            model_name=request.model_name,
            messages=request.messages,
            **kwargs
        )
        
        return ChatResponse.from_model_response(response)
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

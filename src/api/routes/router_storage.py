from fastapi import APIRouter, HTTPException
from ...models.embeddings import embedding_generator
from ...core.storage.database import db
from ...models.model_manager import model_manager
from ...core.chunks.chunk_handler import chunk_handler
from ...entity.Class import CompareEmbeddingsRequest, EmbeddingRequest, EmbeddingResponse, ChunkRequest, ChunkResponse, CompareEmbeddingsResponse, StoreEmbeddingRequest, StoreEmbeddingResponse, SearchSimilarEmbeddingsRequest, SearchSimilarEmbeddingsResponse, RAGRequest, RAGResponse, SimilarEmbedding
from src.core.storage.vector_database import VectorDatabase
import logging
import traceback

vector_db = VectorDatabase('./db/localv.db')

logger = logging.getLogger(__name__)

router_storage = APIRouter()

@router_storage.post("/store_embedding", response_model=StoreEmbeddingResponse)
async def store_embedding(request: StoreEmbeddingRequest):
    try:
        embedding = embedding_generator.generate_embedding(request.text)
        embedding_id = vector_db.add_embedding(embedding, request.metadata)
        return StoreEmbeddingResponse(embedding_id=embedding_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_storage.post("/search_similar_embeddings", response_model=SearchSimilarEmbeddingsResponse)
async def search_similar_embeddings(request: SearchSimilarEmbeddingsRequest):
    try:
        query_embedding = embedding_generator.generate_embedding(request.text)
        similar_embeddings = vector_db.search_similar(query_embedding, request.top_k)
        return SearchSimilarEmbeddingsResponse(similar_embeddings=similar_embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from ...models.embeddings import embedding_generator
from ...models.model_manager import model_manager
from ...core.chunks.chunk_handler import chunk_handler
from ...entity.Class import CompareEmbeddingsRequest, Message, Session, ProcessFileResponse, EmbeddingRequest, EmbeddingResponse, ChunkRequest, ChunkResponse, CompareEmbeddingsResponse, StoreEmbeddingRequest, StoreEmbeddingResponse, SearchSimilarEmbeddingsRequest, SearchSimilarEmbeddingsResponse, RAGRequest, RAGResponse, SimilarEmbedding
from ...core.storage.vector_database import VectorDatabase
from ...core.utils import extract_text_from_document
import logging

vector_db = VectorDatabase('./db/localv.db')

logger = logging.getLogger(__name__)

router_storage = APIRouter()

@router_storage.post("/store_embedding", response_model=StoreEmbeddingResponse)
async def store_embedding(request: StoreEmbeddingRequest):
    try:
        embedding = embedding_generator.generate_embedding(request.text)
        metadata = request.metadata
        
        if request.session is not None:
            kwargs = request.session.dict()
            if kwargs.get("userId"):
                metadata["userId"] = kwargs.get("userId")
            if kwargs.get("sessionId"):
                metadata["sessionId"] = kwargs.get("sessionId")
                
        embedding_id = vector_db.add_embedding(embedding, metadata)
        return StoreEmbeddingResponse(embedding_id=embedding_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_storage.post("/search_similar_embeddings", response_model=SearchSimilarEmbeddingsResponse)
async def search_similar_embeddings(request: SearchSimilarEmbeddingsRequest):
    try:
        query_embedding = embedding_generator.generate_embedding(request.text)
        filter_condition = {}
        
        if request.session is not None:
            if request.session["userId"]:
                filter_condition["userId"] = filter_condition["userId"]
            if request.session["sessionId"]:
                filter_condition["sessionId"] = request.session["sessionId"]
                
        similar_embeddings = vector_db.search_similar(query_embedding, request.top_k, filter_condition)
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
        query_embedding = embedding_generator.generate_embedding(request.query)
        
        filter_condition = {}
        if request.session is not None:
            if request.session["userId"]:
                filter_condition["userId"] = request.session["userId"]
            if request.session["sessionId"]:
                filter_condition["sessionId"] = request.session["sessionId"]
        
        similar_embeddings = vector_db.search_similar(query_embedding, request.top_k, filter_condition)
        logger.debug(f"Found {len(similar_embeddings)} similar embeddings")
        
        context = "\n".join([emb.get('metadata', {}).get('content', '') for emb in similar_embeddings])
        
        prompt = f"""Use the following context to answer the question. If the answer is not contained in the context, say "I don't have enough information to answer this question."

        Context:
        {context}

        Question: {request.query}

        Answer:"""
        
        kwargs = {}
        
        if request.session is not None:
            if request.session["userId"]:
                kwargs.setdefault("session", {})["userId"] = request.session["userId"]
            if request.session["sessionId"]:
                kwargs.setdefault("session", {})["sessionId"] = request.session["sessionId"]
        
        messages: List[Message] = [Message(role="user", content=prompt)]
        
        response = model_manager.generate_chat(
            model_name=request.model_name,
            messages=messages,
            max_tokens=None,
            temperature=0.7,
            **kwargs
        )
        
        # Log the raw response for debugging
        logger.debug(f"Raw model response: {response}")
        
        # 6. Extract the answer from the response
        if isinstance(response, dict):
            if "message" in response and isinstance(response["message"], dict):
                answer = response["message"].get("content", "")
            elif "choices" in response and isinstance(response["choices"], list) and len(response["choices"]) > 0:
                answer = response["choices"][0].get("message", {}).get("content", "")
            else:
                answer = str(response)
        elif isinstance(response, str):
            answer = response
        else:
            logger.error(f"Unexpected response format from model: {response}")
            answer = "Error: Unexpected response format from model."
        
        # 7. Return the answer and the sources
        sources = [
            SimilarEmbedding(
                id=emb.get('id', ''),
                metadata=emb.get('metadata', {}),
                cosine_similarity=emb.get('cosine_similarity', 0.0)
            ) for emb in similar_embeddings
        ]
        return RAGResponse(answer=answer, sources=sources)
    except Exception as e:
        logger.error(f"Error in RAG query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router_storage.post("/process_file", response_model=ProcessFileResponse)
async def process_file( file: UploadFile = File(...), chunk_size: int = 1000, overlap: int = 200, model_name: str = "default_embedding_model", session: Optional[Session] = None ):
    try:
        content = await file.read()
        text = extract_text_from_document(content, file.filename)
        chunks = chunk_handler.process_chunks(text, 'text', chunk_size=chunk_size, overlap=overlap)
        
        embedding_ids = []
        for i, chunk in enumerate(chunks):
            embedding = embedding_generator.generate_embedding(chunk, model_name)
            metadata = {
                "filename": file.filename,
                "content": chunk,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_size": chunk_size,
                "overlap": overlap
            }
            
            if session is not None:
                if session["userId"]:
                    metadata["userId"] = session["userId"]
                if session["sessionId"]:
                    metadata["sessionId"] = session["sessionId"]
            embedding_id = vector_db.add_embedding(embedding, metadata)
            embedding_ids.append(embedding_id)
        
        return ProcessFileResponse(
            filename=file.filename,
            total_chunks=len(chunks),
            embedding_ids=embedding_ids
        )
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

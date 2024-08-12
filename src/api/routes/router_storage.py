from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from ...models.embeddings import embedding_generator
from ...models.model_manager import model_manager
from ...core.chunks.chunk_handler import chunk_handler
from ...entity.Class import CompareEmbeddingsRequest, GetEmbeddingResponse, SyncResponse, ListEmbeddingsResponse, Message, RequestBasic, Session, ProcessFileResponse, EmbeddingRequest, EmbeddingResponse, ChunkRequest, ChunkResponse, CompareEmbeddingsResponse, StoreEmbeddingRequest, StoreEmbeddingResponse, SearchSimilarEmbeddingsRequest, SearchSimilarEmbeddingsResponse, RAGRequest, RAGResponse, SimilarEmbedding
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
            if request.session.userId:
                metadata["userId"] = request.session.userId
            if request.session.sessionId:
                metadata["sessionId"] = request.session.sessionId
                
        embedding_id = vector_db.add_embedding(embedding, metadata)
        return StoreEmbeddingResponse(embedding_id=embedding_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router_storage.post("/list_embeddings", response_model=ListEmbeddingsResponse)
async def list_embeddings(request: RequestBasic):
    try:
        filters = {}
        
        if request.session:
            if request.session.userId:
                filters["userId"] = request.session.userId
            if request.session.sessionId:
                filters["sessionId"] = request.session.sessionId
        embeddings = vector_db.list_embeddings(user_id=filters["userId"], session_id=filters["sessionId"])
        return ListEmbeddingsResponse(embeddings=embeddings)
    except Exception as e:
        logger.error(f"Error listing embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router_storage.post("/search_similar_embeddings", response_model=SearchSimilarEmbeddingsResponse)
async def search_similar_embeddings(request: SearchSimilarEmbeddingsRequest):
    try:
        query_embedding = embedding_generator.generate_embedding(request.text)
        filter_condition = {}
        
        if request.cosine_similarity:
            filter_condition["cosine_similarity"] = request.cosine_similarity
        
        if request.session:
            if request.session.userId:
                filter_condition["userId"] = request.session.userId
            if request.session.sessionId:
                filter_condition["sessionId"] = request.session.sessionId
                
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
            if request.session.userId:
                filter_condition["userId"] = request.session.userId
            if request.session.sessionId:
                filter_condition["sessionId"] = request.session.sessionId
        
        similar_embeddings = vector_db.search_similar(query_embedding, request.top_k, filter_condition)
        logger.debug(f"Found {len(similar_embeddings)} similar embeddings")
        
        context = "\n".join([emb.get('metadata', {}).get('content', '') for emb in similar_embeddings])
        
        prompt = f"""Use the following context to answer the question. If the answer is not contained in the context, say "I don't have enough information to answer this question."

        Context:
        {context}

        Question: {request.query}

        Answer:"""
        
        messages: List[Message] = [Message(role="user", content=prompt)]
        
        response = model_manager.generate_chat(
            model_name=request.model_name,
            messages=messages,
            max_tokens=None,
            temperature=0.7,
            **filter_condition
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
                if session.userId:
                    metadata["userId"] = session.userId
                if session.sessionId:
                    metadata["sessionId"] = session.sessionId
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


@router_storage.get("/get_embedding", response_model=GetEmbeddingResponse)
async def get_embedding(embedding_id: str, user_id: Optional[str] = None, session_id: Optional[str] = None):
    try:
        embedding = vector_db.get_embedding_by_id(embedding_id, user_id=user_id, session_id=session_id)
        if embedding is None:
            raise HTTPException(status_code=404, detail="Embedding not found")
        return GetEmbeddingResponse(embedding=embedding)
    except Exception as e:
        logger.error(f"Error getting embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router_storage.post("/sync_to_firebase", response_model=SyncResponse)
async def sync_to_firebase():
    try:
        vector_db.sync_to_firebase()
        return SyncResponse(message="Sync to Firebase completed successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing to Firebase: {str(e)}")

@router_storage.post("/sync_from_firebase", response_model=SyncResponse)
async def sync_from_firebase(request: RequestBasic):
    try:
        vector_db.sync_from_firebase(request)
        return SyncResponse(message="Sync from Firebase completed successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing from Firebase: {str(e)}")

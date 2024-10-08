import base64
import json
import chromadb
from chromadb.config import Settings
import time
from typing import List, Dict, Any, Optional
from src.entity.Class import RequestBasic
from src.core.storage.firebase import firebase_connection
from src.core.storage.firebase_sync.sync import sync_to_firebase, sync_from_firebase
from src.core.config import settings
from src.core.utils import setup_logger

logger = setup_logger(__name__)

class VectorDatabase:
    def __init__(self, persist_directory: str):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("embeddings")
        self.sync_queue = []

    def add_embedding(self, embedding: List[float], metadata: Dict[str, Any]) -> str:
        timestamp = int(time.time())
        id = None
        if metadata.get('id'):
            id = metadata['id']
        else:
            id = f"emb_{timestamp}"
        
        # Add to local ChromaDB
        self.collection.add(
            embeddings=[embedding],
            metadatas=[{**metadata, "last_updated": timestamp}],
            ids=[id]
        )
        
        # Try to sync immediately with Firebase
        if self.check_connection() and metadata.get('id') is None:
            try:
                self._sync_to_firebase(id, embedding, metadata, timestamp)
                logger.info(f"Embedding {id} synced to Firebase immediately")
            except Exception as e:
                logger.error(f"Error syncing embedding {id} to Firebase: {str(e)}")
                self.sync_queue.append(('add', id, embedding, metadata, timestamp))
        else:
            self.sync_queue.append(('add', id, embedding, metadata, timestamp))
            logger.info(f"Embedding {id} queued for later sync to Firebase")
        
        return id

    def _sync_to_firebase(self, id: str, embedding: List[float], metadata: Dict[str, Any], timestamp: int):
        # Determine the base path for both Storage and Firestore
        if 'userId' in metadata and 'sessionId' in metadata:
            base_path = f"{settings.ROOTCOLECCTION}/{metadata['userId']}/{metadata['sessionId']}/{id}"
        elif 'userId' in metadata:
            base_path = f"{settings.ROOTCOLECCTION}/{metadata['userId']}/embeddings/{id}"
        else:
            base_path = f"{settings.ROOTCOLECCTION}/embeddings/root/{id}"

        # Storage path (same as base_path)
        storage_path = f"{base_path}.json"

        # Save embedding to Firebase Storage
        embedding_json = json.dumps(embedding)
        embedding_bytes = embedding_json.encode('utf-8')
        embedding_base64 = base64.b64encode(embedding_bytes).decode('utf-8')
        
        storage_result = firebase_connection.upload_to_storage(storage_path, embedding_base64)
        
        if storage_result is None:
            raise Exception(f"Failed to upload embedding {id} to Firebase Storage")

        # Firestore path (same as base_path)
        firestore_path = base_path
        
        metadata['id'] = id
        firestore_data = {
            'metadata': metadata,
            'type': 'embedding',
            'last_updated': timestamp,
            'embedding_ref': storage_path  # Reference to the Storage file
        }
        
        firestore_result = firebase_connection.add_document(firestore_path, firestore_data)
        
        if firestore_result is None:
            raise Exception(f"Failed to sync metadata for embedding {id} to Firestore")
        

    def search_similar(self, query_embedding: List[float], top_k: int = 5, filter_condition: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        try:
            where_clause = {}
            similarity_threshold = 0.0

            if filter_condition:
                conditions = []

                if 'cosine_similarity' in filter_condition:
                    similarity_threshold = filter_condition.pop('cosine_similarity', 0.0)
                
                for key, value in filter_condition.items():
                    if isinstance(value, dict):
                        operator, match_value = next(iter(value.items()))
                        conditions.append({key: {operator: match_value}})
                    else:
                        conditions.append({key: {"$eq": value}})

                if conditions and len(conditions) > 1:
                    where_clause = {"$and": conditions}

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                where=where_clause if where_clause else None,
                include=["metadatas", "distances", "documents"]
            )
            
            similar_embeddings = []
            for i in range(len(results['ids'][0])):
                cosine_similarity = 1 - results['distances'][0][i]
                if cosine_similarity >= similarity_threshold:
                    similar_embeddings.append({
                        'id': results['ids'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'content': results['documents'][0][i],
                        'cosine_similarity': cosine_similarity
                    })

            return similar_embeddings

            # similar_embeddings = [
            #     {
            #         'id': results['ids'][0][i],
            #         'metadata': results['metadatas'][0][i],
            #         'content': results['documents'][0][i],
            #         'cosine_similarity': 1 - results['distances'][0][i]
            #     }
            #     for i in range(len(results['ids'][0]))
            # ]

            # return similar_embeddings

        except Exception as e:
            logger.error(f"Error searching similar embeddings: {str(e)}", exc_info=True)
            return []


    def sync_to_firebase(self):
        while self.sync_queue:
            action, *args = self.sync_queue.pop(0)
            try:
                if action == 'add':
                    sync_to_firebase(*args)
                # Add more actions as needed
            except Exception as e:
                logger.error(f"Error syncing to Firebase: {str(e)}")
                self.sync_queue.append((action, *args))

    def sync_from_firebase(self, filters: RequestBasic = None):
        try:
            # Obtener embeddings de Firebase
            existing_ids = set(self.collection.get()['ids'])
            
            synced_embeddings = sync_from_firebase(filters, existing_ids)
            
            # Obtener los IDs de embeddings existentes en la base de datos local
            
            embeddings_to_add = []
            for emb in synced_embeddings:
                emb_id = emb['metadata'].get('id')
                if emb_id and emb_id.startswith('emb_') and emb_id not in existing_ids:
                    embeddings_to_add.append(emb)
            
            # Añadir solo los embeddings que faltan
            for emb in embeddings_to_add:
                self.add_embedding(emb['embedding'], emb['metadata'])
            
            logger.info(f"Synced {len(embeddings_to_add)} new embeddings from Firebase")
        except Exception as e:
            logger.error(f"Error syncing from Firebase: {str(e)}", exc_info=True)
        

    def _was_synced(self, sync_item):
        # Implement logic to check if the item was successfully synced
        # For now, we'll assume it wasn't synced to be safe
        return False
    

    # def get_last_local_update(self) -> int:
    #     try:
    #         collection_count = self.collection.count()
    #         if collection_count == 0:
    #             logger.info("Collection is empty, returning 0 as last update time")
    #             return 0

    #         # Request just one result, ordered by last_updated in descending order
    #         results = self.collection.query(
    #             query_embeddings=[[0] * 384],  # Dummy embedding
    #             n_results=1,
    #             where={},
    #             include=["metadatas"]
    #         )
            
    #         if results['metadatas'] and results['metadatas'][0]:
    #             last_updated = results['metadatas'][0][0].get('last_updated', 0)
    #             logger.info(f"Last local update time: {last_updated}")
    #             return last_updated
            
    #         logger.warning("No metadata found in the collection")
    #         return 0
    #     except Exception as e:
    #         logger.error(f"Error getting last local update: {str(e)}")
    #         return 0

    def check_connection(self) -> bool:
        # Implement real connection check logic here
        # For now, we'll assume there's always a connection
        return True
    

    def delete_embedding(self, embedding_id: str) -> bool:
        try:
            self.collection.delete(ids=[embedding_id])
            logger.info(f"Deleted embedding with id: {embedding_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting embedding: {str(e)}")
            return False
        

    def update_embedding(self, embedding_id: str, new_embedding: List[float], new_metadata: Dict[str, Any]) -> bool:
        try:
            self.collection.update(
                ids=[embedding_id],
                embeddings=[new_embedding],
                metadatas=[new_metadata]
            )
            logger.info(f"Updated embedding with id: {embedding_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating embedding: {str(e)}")
            return False
        

    def store_document(self, content: str, metadata: Dict[str, Any]) -> str:
        timestamp = int(time.time())
        id = f"doc_{timestamp}"
        
        self.collection.add(
            documents=[content],
            metadatas=[{**metadata, "last_updated": timestamp}],
            ids=[id]
        )
        
        self.sync_queue.append(('add_document', id, content, metadata, timestamp))
        
        return id
    

    def get_document_by_id(self, document_id: str) -> Dict[str, Any]:
        results = self.collection.get(
            ids=[document_id],
            include=["metadatas", "documents"]
        )
        
        if results['ids']:
            return {
                "id": results['ids'][0],
                "content": results['documents'][0],
                "metadata": results['metadatas'][0]
            }
        else:
            logger.warning(f"Document with id {document_id} not found")
            return None
        

    def update_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        try:
            self.collection.update(
                ids=[document_id],
                documents=[content],
                metadatas=[metadata]
            )
            logger.info(f"Updated document with id: {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False
    
    def list_embeddings(self, user_id: str = None, session_id: str = None) -> List[Dict[str, Any]]:
        try:
            where_clause = {}
            conditions = []

            if user_id:
                conditions.append({'userId': {"$eq": user_id}})
            if session_id:
                conditions.append({'sessionId': {"$eq": session_id}})
            
            if conditions and len(conditions) > 1:
                where_clause = {"$and": conditions}

            results = self.collection.query(
                query_embeddings=[[0] * 384],  # Dummy embedding
                n_results=self.collection.count(),
                where=where_clause,
                include=["metadatas", "documents"]
            )

            embeddings = []
            for i in range(len(results['ids'][0])):
                embeddings.append({
                    'id': results['ids'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'content': results['documents'][0][i]
                })
            return embeddings
        except Exception as e:
            logger.error(f"Error listing embeddings: {str(e)}", exc_info=True)
            return []

    
    def get_embedding_by_id(self, embedding_id: str, user_id: str = None, session_id: str = None) -> Dict[str, Any]:
        try:
            where_clause = {"id": {"$eq": embedding_id}}
            if user_id:
                where_clause['userId'] = {"$eq": user_id}
            if session_id:
                where_clause['sessionId'] = {"$eq": session_id}

            results = self.collection.query(
                query_embeddings=[[0] * 384],  # Dummy embedding
                n_results=1,
                where=where_clause,
                include=["metadatas", "documents"]
            )

            if results['ids']:
                return {
                    "id": results['ids'][0],
                    "metadata": results['metadatas'][0],
                    "content": results['documents'][0]
                }
            else:
                logger.warning(f"Embedding with id {embedding_id} not found")
                return None
        except Exception as e:
            logger.error(f"Error getting embedding by id: {str(e)}", exc_info=True)
            return None

# Uso
vector_db = VectorDatabase('./db/localv.db')
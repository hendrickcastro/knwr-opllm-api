import chromadb
from chromadb.config import Settings
import time
from typing import List, Dict, Any
from src.core.storage.firebase import firebase_connection
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
        id = f"emb_{timestamp}"
        
        # Add to local ChromaDB
        self.collection.add(
            embeddings=[embedding],
            metadatas=[{**metadata, "last_updated": timestamp}],
            ids=[id]
        )
        
        # Try to sync immediately with Firestore
        if self.check_connection():
            try:
                self._sync_to_firestore(id, embedding, metadata, timestamp)
                logger.info(f"Embedding {id} synced to Firestore immediately")
            except Exception as e:
                logger.error(f"Error syncing embedding {id} to Firestore: {str(e)}")
                self.sync_queue.append(('add', id, embedding, metadata, timestamp))
        else:
            self.sync_queue.append(('add', id, embedding, metadata, timestamp))
            logger.info(f"Embedding {id} queued for later sync to Firestore")
        
        return id
    

    def _sync_to_firestore(self, id: str, embedding: List[float], metadata: Dict[str, Any], timestamp: int):
        firebase_path = f'{settings.ROOTCOLECCTION}/embeddings'
        if 'userId' in metadata:
            firebase_path += f'/{metadata["userId"]}'
        if 'sessionId' in metadata:
            firebase_path += f'/{metadata["sessionId"]}'
        firebase_path += f'/{id}'
        
        firebase_result = firebase_connection.add_document(firebase_path, {
            'embedding': embedding,
            'metadata': metadata,
            'last_updated': timestamp
        })
        
        if firebase_result is None:
            raise Exception(f"Failed to sync embedding {id} to Firestore")
        

    def search_similar(self, query_embedding: List[float], top_k: int = 5, filter_condition: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        try:
            where_clause = {}
            if filter_condition:
                for key, value in filter_condition.items():
                    where_clause[key] = {"$eq": value}
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, self.collection.count()),
                where=where_clause,
                include=["metadatas", "distances", "documents"]
            )
            
            similar_embeddings = []
            for i in range(len(results['ids'][0])):
                similar_embeddings.append({
                    'id': results['ids'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'content': results['documents'][0][i],
                    'cosine_similarity': 1 - results['distances'][0][i]
                })
            
            return similar_embeddings
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {str(e)}", exc_info=True)
            return []
        

    def sync_with_cloud(self):
        if not self.check_connection():
            logger.warning("No internet connection. Syncing queued for later.")
            return

        for action, id, embedding, metadata, timestamp in self.sync_queue:
            if action == 'add':
                try:
                    self._sync_to_firestore(id, embedding, metadata, timestamp)
                    logger.info(f"Queued embedding {id} synced to Firestore")
                except Exception as e:
                    logger.error(f"Error syncing queued embedding {id} to Firestore: {str(e)}")
                    continue  # Keep this item in the queue

        self.sync_queue = [item for item in self.sync_queue if not self._was_synced(item)]

        self._sync_from_cloud()
        

    def _was_synced(self, sync_item):
        # Implement logic to check if the item was successfully synced
        # For now, we'll assume it wasn't synced to be safe
        return False
    

    def _sync_from_cloud(self):
        last_local_update = self.get_last_local_update()
        
        try:
            # Sync from Firestore
            firestore_updates = firebase_connection.db.collection(f'{settings.ROOTCOLECCTION}/embeddings').where('last_updated', '>', last_local_update).get()
            for doc in firestore_updates:
                data = doc.to_dict()
                self.add_embedding(data['embedding'], data['metadata'])
            logger.info(f"Synced {len(firestore_updates)} embeddings from Firestore")
        except Exception as e:
            logger.error(f"Error syncing from Firestore: {str(e)}")
            

    def get_last_local_update(self) -> int:
        results = self.collection.query(
            query_embeddings=[[0] * 384],  # Dummy embedding
            n_results=1,
            where={"last_updated": {"$exists": True}},
            include=["metadatas"]
        )
        if results['metadatas'] and results['metadatas'][0]:
            return results['metadatas'][0][0].get('last_updated', 0)
        return 0
    

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
        

# Uso
vector_db = VectorDatabase('./db/localv.db')
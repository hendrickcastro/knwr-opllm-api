import chromadb
from chromadb.config import Settings
import time
from typing import List, Dict, Any
from src.core.storage.database import db as mongodb_client
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
        id = f"emb_{timestamp}"  # Genera un ID único
        
        self.collection.add(
            embeddings=[embedding],
            metadatas=[{**metadata, "last_updated": timestamp}],
            ids=[id]
        )
        
        self.sync_queue.append(('add', id, embedding, metadata, timestamp))
        
        return id

    def search_similar(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"]
        )
        
        return [
            {
                'id': id,
                'metadata': metadata,
                'cosine_similarity': 1 - distance  # ChromaDB usa distancia, la convertimos a similitud
            }
            for id, metadata, distance in zip(results['ids'][0], results['metadatas'][0], results['distances'][0])
        ]

    def sync_with_cloud(self):
        if not self.check_connection():
            logger.warning("No internet connection. Syncing queued for later.")
            return

        # Sync from local to cloud
        for action, id, embedding, metadata, timestamp in self.sync_queue:
            if action == 'add':
                try:
                    # Sync to MongoDB
                    mongodb_result = mongodb_client.store_embedding(embedding, metadata)
                    if mongodb_result is None:
                        logger.warning(f"Failed to sync embedding {id} to MongoDB")
                    
                    # Sync to Firebase
                    firebase_result = firebase_connection.add_document(f'{settings.ROOTCOLECCTION}/embeddings/{id}', {
                        'embedding': embedding,
                        'metadata': metadata,
                        'last_updated': timestamp
                    })
                    if firebase_result is None:
                        logger.warning(f"Failed to sync embedding {id} to Firebase")
                    
                    if mongodb_result is not None or firebase_result is not None:
                        logger.info(f"Synced embedding {id} to cloud storage")
                    else:
                        logger.warning(f"Failed to sync embedding {id} to any cloud storage")
                except Exception as e:
                    logger.error(f"Error syncing embedding {id} to cloud: {str(e)}")

        self.sync_queue = [item for item in self.sync_queue if not self._was_synced(item)]

        # Sync from cloud to local
        self._sync_from_cloud()

    def _was_synced(self, sync_item):
        # Por ahora, asumimos que nada se sincronizó con éxito
        return False

    def _sync_from_cloud(self):
        last_local_update = self.get_last_local_update()
        
        # Sync from MongoDB
        try:
            mongo_updates = mongodb_client.db.embeddings.find({'last_updated': {'$gt': last_local_update}})
            for doc in mongo_updates:
                self.add_embedding(doc['embedding'], doc['metadata'])
        except Exception as e:
            logger.error(f"Error syncing from MongoDB: {str(e)}")

        # Sync from Firebase
        try:
            firestore_updates = firebase_connection.db.collection('embeddings').where('last_updated', '>', last_local_update).get()
            for doc in firestore_updates:
                data = doc.to_dict()
                self.add_embedding(data['embedding'], data['metadata'])
        except Exception as e:
            logger.error(f"Error syncing from Firebase: {str(e)}")

    def get_last_local_update(self) -> int:
        results = self.collection.query(
            query_embeddings=[[0] * 384],  # Embedding ficticio
            n_results=1,
            where={"last_updated": {"$exists": True}},
            include=["metadatas"]
        )
        if results['metadatas'] and results['metadatas'][0]:
            return results['metadatas'][0][0].get('last_updated', 0)
        return 0

    def check_connection(self) -> bool:
        # Implementa una lógica real de verificación de conexión aquí
        return True  # Por ahora, asumimos que siempre hay conexión

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
        # Este método ahora usa ChromaDB en lugar de MongoDB directamente
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
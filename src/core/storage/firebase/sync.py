import base64
import json
from typing import Dict, Any, List
from src.core.storage.firebase import firebase_connection
from src.core.config import settings
from src.core.utils import setup_logger

logger = setup_logger(__name__)

def sync_to_firebase(id: str, embedding: List[float], metadata: Dict[str, Any], timestamp: int):
    base_path = determine_base_path(metadata)
    storage_path = f"{base_path}.json"
    firestore_path = base_path

    # Save embedding to Firebase Storage
    embedding_base64 = encode_embedding(embedding)
    storage_result = firebase_connection.upload_to_storage(storage_path, embedding_base64)
    
    if storage_result is None:
        raise Exception(f"Failed to upload embedding {id} to Firebase Storage")

    # Save metadata to Firestore
    firestore_data = {
        'metadata': metadata,
        'type': 'embedding',
        'last_updated': timestamp,
        'embedding_ref': storage_path
    }
    
    firestore_result = firebase_connection.add_document(firestore_path, firestore_data)
    
    if firestore_result is None:
        raise Exception(f"Failed to sync metadata for embedding {id} to Firestore")

def sync_from_firebase(last_local_update: int) -> List[Dict[str, Any]]:
    synced_embeddings = []
    base_ref = firebase_connection.db.collection(settings.ROOTCOLECCTION)
    
    def search_embeddings(ref):
        for doc in ref.get():
            if doc.id.startswith('emb_'):
                data = doc.to_dict()
                if data['last_updated'] > last_local_update:
                    embedding_ref = data.get('embedding_ref')
                    if embedding_ref:
                        embedding_base64 = firebase_connection.download_from_storage(embedding_ref)
                        if embedding_base64:
                            embedding = decode_embedding(embedding_base64)
                            synced_embeddings.append({
                                'id': doc.id,
                                'embedding': embedding,
                                'metadata': data['metadata']
                            })
            else:
                search_embeddings(doc.reference.collections())

    search_embeddings(base_ref)
    return synced_embeddings

def determine_base_path(metadata: Dict[str, Any]) -> str:
    if 'userId' in metadata and 'sessionId' in metadata:
        return f"{settings.ROOTCOLECCTION}/{metadata['userId']}/{metadata['sessionId']}"
    elif 'userId' in metadata:
        return f"{settings.ROOTCOLECCTION}/{metadata['userId']}/embeddings"
    else:
        return f"{settings.ROOTCOLECCTION}/embeddings"

def encode_embedding(embedding: List[float]) -> str:
    embedding_json = json.dumps(embedding)
    embedding_bytes = embedding_json.encode('utf-8')
    return base64.b64encode(embedding_bytes).decode('utf-8')

def decode_embedding(embedding_base64: str) -> List[float]:
    embedding_bytes = base64.b64decode(embedding_base64)
    return json.loads(embedding_bytes.decode('utf-8'))
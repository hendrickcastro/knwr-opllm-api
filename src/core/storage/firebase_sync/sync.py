import base64
import json
from typing import Dict, Any, List, Set
from src.entity.Class import RequestBasic
from src.core.storage.firebase import firebase_connection
from src.core.config import settings
from src.core.utils import setup_logger
from google.cloud.exceptions import BadRequest

logger = setup_logger(__name__)

def sync_to_firebase(id: str, embedding: List[float], metadata: Dict[str, Any], timestamp: int):
    base_path = determine_base_path(metadata)
    storage_path = f"{base_path}/{id}.json"
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
    
    
def get_embeddings_from_level(path: str) -> List[Dict[str, Any]]:
    embeddings = []
    try:
        query = (firebase_connection.db.collection(path).where('type', '==', 'embedding'))

        docs = query.get()
        for doc in docs:
            data = doc.to_dict()
            embeddings.append(data)
    except BadRequest as e:
        if "The query requires an index" in str(e):
            logger.error(f"Index required for query on path {path}. Please create the index in Firebase console.")
            logger.error(f"Error details: {str(e)}")
        else:
            raise
    return embeddings

def sync_from_firebase(filters: RequestBasic = None, ids: Set[int] = None) -> List[Dict[str, Any]]:
    all_embeddings = []
    
    # Nivel superior (root)
    all_embeddings.extend(get_embeddings_from_level(f"{settings.ROOTCOLECCTION}/embeddings/root"))
    
    if filters.session:
        # Nivel de usuario
        session = filters.session
        if session.userId:
            all_embeddings.extend(get_embeddings_from_level(f"{settings.ROOTCOLECCTION}/{session.userId}/embeddings"))
        
        if session.sessionId and session.userId:
            all_embeddings.extend(get_embeddings_from_level(f"{settings.ROOTCOLECCTION}/{session.userId}/{session.sessionId}"))
    
    ## filter exclude ids
    if ids:
        all_embeddings = [emb for emb in all_embeddings if emb['metadata'].get('id') not in ids]
        
    synced_embeddings = []
    for emb_data in all_embeddings:
        embedding_ref = emb_data.get('embedding_ref')
        if embedding_ref:
            embedding_base64 = firebase_connection.download_from_storage(embedding_ref)
            if embedding_base64:
                embedding = decode_embedding(embedding_base64)
                synced_embeddings.append({
                    'embedding': embedding,
                    'metadata': emb_data['metadata']
                })
                logger.info(f"Synced embedding {emb_data['metadata'].get('id')} from Firebase")
            else:
                logger.warning(f"Failed to download embedding from {embedding_ref}")
        else:
            logger.warning(f"No embedding reference found for {emb_data['metadata'].get('id')}")
    
    return synced_embeddings

def determine_base_path(metadata: Dict[str, Any]) -> str:
    if 'userId' in metadata and 'sessionId' in metadata:
        return f"{settings.ROOTCOLECCTION}/{metadata['userId']}/{metadata['sessionId']}"
    elif 'userId' in metadata:
        return f"{settings.ROOTCOLECCTION}/{metadata['userId']}/embeddings"
    else:
        return f"{settings.ROOTCOLECCTION}/embeddings/root"

def encode_embedding(embedding: List[float]) -> str:
    embedding_json = json.dumps(embedding)
    embedding_bytes = embedding_json.encode('utf-8')
    return base64.b64encode(embedding_bytes).decode('utf-8')

def decode_embedding(embedding_base64: str) -> List[float]:
    embedding_bytes = base64.b64decode(embedding_base64)
    return json.loads(embedding_bytes.decode('utf-8'))
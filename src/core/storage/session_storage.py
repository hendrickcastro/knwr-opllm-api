from typing import Dict, Any
import json
import os
from .firebase import firebase_connection
from ...core.config import settings
from ...core.utils import setup_logger

logger = setup_logger(__name__)

class SessionStorage:
    def __init__(self, local_storage_path: str):
        self.local_storage_path = local_storage_path
        self.ensure_local_storage()

    def ensure_local_storage(self):
        if not os.path.exists(self.local_storage_path):
            os.makedirs(self.local_storage_path)

    def store_locally(self, user_id: str, session_id: str, data: Dict[str, Any]):
        file_path = os.path.join(self.local_storage_path, f"{user_id}_{session_id}.json")
        with open(file_path, 'w') as f:
            json.dump(data, f)

    def store_in_firebase(self, user_id: str, session_id: str, data: Dict[str, Any]):
        doc_id = firebase_connection.add_document(
            f'{settings.ROOTCOLECCTION}/{user_id}/{session_id}',
            data
        )
        return doc_id

    def sync_with_firebase(self):
        for filename in os.listdir(self.local_storage_path):
            if filename.endswith('.json'):
                user_id, session_id = filename[:-5].split('_')
                file_path = os.path.join(self.local_storage_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                try:
                    self.store_in_firebase(user_id, session_id, data)
                    os.remove(file_path)  # Remove local file after successful sync
                    logger.info(f"Synced and removed local file: {filename}")
                except Exception as e:
                    logger.error(f"Error syncing {filename} to Firebase: {str(e)}")

    def store_session_data(self, user_id: str, session_id: str, data: Dict[str, Any]):
        self.store_locally(user_id, session_id, data)
        try:
            doc_id = self.store_in_firebase(user_id, session_id, data)
            logger.info(f"Stored in Firebase with ID: {doc_id}")
        except Exception as e:
            logger.error(f"Error storing in Firebase: {str(e)}. Data stored locally.")

session_storage = SessionStorage('./local_sessions')
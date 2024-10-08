import os
import json
from typing import Any, Dict
import firebase_admin
from firebase_admin import credentials, firestore, storage
import threading
import time

class FirebaseConnection:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FirebaseConnection, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance

    def _initialize(self):
        self.app = None
        self.db = None
        self.bucket = None
        self.config_ref = None
        self.local_config = {}
        self.connect()
        
    def connect(self):
        retry_count = 0
        max_retries = 5
        while retry_count < max_retries:
            try:
                if not firebase_admin._apps:
                    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'configdb.json')
                    with open(config_path, 'r') as config_file:
                        config = json.load(config_file)

                    cred = credentials.Certificate(config)
                    self.app = firebase_admin.initialize_app(cred, {
                        'storageBucket': f"{config['project_id']}.appspot.com"
                    })
                
                self.db = firestore.client()
                self.bucket = storage.bucket()
                self.config_ref = self.db.collection('config').document('app_config')
                self.load_config()
                self.start_listener()
                print("Firebase connection established successfully")
                return
            except Exception as e:
                retry_count += 1
                print(f"Failed to connect to Firebase (attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count == max_retries:
                    print("Max retries reached. Unable to connect to Firebase.")
                    self.app = None
                    self.db = None
                    self.bucket = None
                else:
                    time.sleep(2 ** retry_count)  # Exponential backoff
                    
    def _ensure_connection(self):
        if self.app is None or self.db is None:
            self.connect()
        if self.app is None or self.db is None:
            raise Exception("No Firebase connection available")

    def load_config(self):
        doc = self.config_ref.get()
        if doc.exists:
            self.local_config = doc.to_dict()
        else:
            self.local_config = {}

    def get(self, key: str, default: Any = None) -> Any:
        try:
            self._ensure_connection()
            return self.local_config.get(key, default)
        except Exception as e:
            print(f"Error adding document to Firebase: {str(e)}")
            return None  # Return None instead of raising an exception

    def start_listener(self):
        def on_snapshot(doc_snapshot, changes, read_time):
            for doc in doc_snapshot:
                self.local_config = doc.to_dict()
                print("Configuration updated from Firebase")

        self.config_ref.on_snapshot(on_snapshot)

    def add_document(self, collection_path: str, data: Dict[str, Any]) -> str:
        try:
            self._ensure_connection()
            # Dividimos la ruta en sus componentes
            path_parts = collection_path.split('/')
            
            # Comenzamos con la referencia a la base de datos
            current_ref = self.db
            
            # Iteramos a través de las partes de la ruta
            for i, part in enumerate(path_parts):
                if i % 2 == 0:
                    # Es una colección
                    current_ref = current_ref.collection(part)
                else:
                    # Es un documento
                    current_ref = current_ref.document(part)
            
            # Añadimos los datos al último nivel
            if isinstance(current_ref, firestore.CollectionReference):
                # Si terminamos en una colección, añadimos un nuevo documento
                doc_ref = current_ref.add(data)
                return doc_ref[1].id
            else:
                # Si terminamos en un documento, establecemos los datos
                current_ref.set(data)
                return current_ref.id
            
        except Exception as e:
            print(f"Error adding document to Firebase: {str(e)}")
            return None  # Return None instead of raising an exception
        
    def add_document_with_id(self, collection_path: str, doc_id: str, data: Dict[str, Any]) -> str:
        try:
            self._ensure_connection()
            # Dividimos la ruta en sus componentes
            path_parts = collection_path.split('/')
            
            # Comenzamos con la referencia a la base de datos
            current_ref = self.db
            
            # Iteramos a través de las partes de la ruta
            for i, part in enumerate(path_parts):
                if i % 2 == 0:
                    # Es una colección
                    current_ref = current_ref.collection(part)
                else:
                    # Es un documento
                    current_ref = current_ref.document(part)
            
            # Añadimos los datos al último nivel usando el doc_id
            if isinstance(current_ref, firestore.CollectionReference):
                # Si terminamos en una colección, establecemos el documento con el doc_id
                current_ref.document(doc_id).set(data)
                return doc_id
            else:
                # Si terminamos en un documento, establecemos los datos (esto debería ser raro)
                current_ref.set(data)
                return current_ref.id
        except Exception as e:
            print(f"Error adding document to Firebase: {str(e)}")
            return None  # Return None instead of raising an exception

    def get_document(self, collection: str, doc_id: str) -> Dict[str, Any]:
        try:
            self._ensure_connection()
            doc_ref = self.db.collection(collection).document(doc_id)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as e:
            print(f"Error adding document to Firebase: {str(e)}")
            return None  # Return None instead of raising an exception

    def update_document(self, collection: str, doc_id: str, data: Dict[str, Any]) -> None:
        try:
            self._ensure_connection()
            doc_ref = self.db.collection(collection).document(doc_id)
            doc_ref.update(data)
        except Exception as e:
            print(f"Error adding document to Firebase: {str(e)}")
            return None  # Return None instead of raising an exception

    def delete_document(self, collection: str, doc_id: str) -> None:
        try:
            self._ensure_connection()
            doc_ref = self.db.collection(collection).document(doc_id)
            doc_ref.delete()
        except Exception as e:
            print(f"Error adding document to Firebase: {str(e)}")
            return None  # Return None instead of raising an exception
        
    def upload_to_storage(self, storage_path: str, data: str) -> bool:
        try:
            self._ensure_connection()
            if not self.bucket:
                raise ValueError("Storage bucket not initialized")
            blob = self.bucket.blob(storage_path)
            blob.upload_from_string(data)
            return True
        except Exception as e:
            print(f"Error uploading to Firebase Storage: {str(e)}")
            return False

    def download_from_storage(self, storage_path: str) -> str:
        try:
            self._ensure_connection()
            if not self.bucket:
                raise ValueError("Storage bucket not initialized")
            blob = self.bucket.blob(storage_path)
            return blob.download_as_text()
        except Exception as e:
            print(f"Error downloading from Firebase Storage: {str(e)}")
            return None

firebase_connection = FirebaseConnection()
import os
import json
from typing import Any, Dict
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import db as realtime_db
import threading

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
        if not firebase_admin._apps:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'configdb.json')
            with open(config_path, 'r') as config_file:
                config = json.load(config_file)

            cred = credentials.Certificate(config)
            firebase_admin.initialize_app(cred)
        
        self.db = firestore.client()
        self.config_ref = self.db.collection('config').document('app_config')
        self.local_config = {}
        self.load_config()
        self.start_listener()

    def load_config(self):
        doc = self.config_ref.get()
        if doc.exists:
            self.local_config = doc.to_dict()
        else:
            self.local_config = {}

    def get(self, key: str, default: Any = None) -> Any:
        return self.local_config.get(key, default)

    def start_listener(self):
        def on_snapshot(doc_snapshot, changes, read_time):
            for doc in doc_snapshot:
                self.local_config = doc.to_dict()
                print("Configuration updated from Firebase")

        self.config_ref.on_snapshot(on_snapshot)

    def add_document(self, collection_path: str, data: Dict[str, Any]) -> str:
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

    def get_document(self, collection: str, doc_id: str) -> Dict[str, Any]:
        doc_ref = self.db.collection(collection).document(doc_id)
        doc = doc_ref.get()
        if doc.exists:
            return doc.to_dict()
        return None

    def update_document(self, collection: str, doc_id: str, data: Dict[str, Any]) -> None:
        doc_ref = self.db.collection(collection).document(doc_id)
        doc_ref.update(data)

    def delete_document(self, collection: str, doc_id: str) -> None:
        doc_ref = self.db.collection(collection).document(doc_id)
        doc_ref.delete()

firebase_connection = FirebaseConnection()
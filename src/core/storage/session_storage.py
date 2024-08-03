import sqlite3
import json
import os
import time
from typing import Dict, Any
from ..config import settings
from ..utils import setup_logger
from .firebase import firebase_connection

logger = setup_logger(__name__)

# Inicialización de la base de datos
def initialize_database(db_path: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Crear tabla para almacenar sesiones con múltiples mensajes y estado de sincronización
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            guid TEXT NOT NULL,
            data TEXT NOT NULL,
            timestamp REAL NOT NULL,
            synced INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

initialize_database(settings.SQLITE_DB_PATH)

class SessionStorage:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def store_locally(self, user_id: str, session_id: str, data: Dict[str, Any]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sessions (user_id, session_id, guid, data, timestamp, synced) 
            VALUES (?, ?, ?, ?, ?, 0)
        ''', (user_id, session_id, data.get("guid"), json.dumps(data), data['timestamp']))
        conn.commit()
        conn.close()

    def store_in_firebase(self, user_id: str, session_id: str, data: Dict[str, Any]):
        ## use data id as document id
        doc_id = data.get("guid")
        if not doc_id:
            raise ValueError("Data must contain a 'guid' field to be used as document ID")
        
        # Establece el id en los datos
        data["id"] = doc_id
        firebase_connection.add_document_with_id(
            f'{settings.ROOTCOLECCTION}/{user_id}/{session_id}',
            doc_id,
            data
        )
        return doc_id

    def sync_with_firebase(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM sessions WHERE synced = 0')
        rows = cursor.fetchall()

        for row in rows:
            user_id, session_id, data, timestamp = row[1], row[2], json.loads(row[3]), row[4]
            try:
                self.store_in_firebase(user_id, session_id, data)
                cursor.execute('UPDATE sessions SET synced = 1 WHERE id = ?', (row[0],))
                conn.commit()
                logger.info(f"Synced and marked local session as synced: {user_id}_{session_id}")
            except Exception as e:
                logger.error(f"Error syncing {user_id}_{session_id} to Firebase: {str(e)}")

        conn.close()

    def store_session_data(self, user_id: str, session_id: str, data: Dict[str, Any]):
        data['timestamp'] = time.time()
        self.store_locally(user_id, session_id, data)
        try:
            doc_id = self.store_in_firebase(user_id, session_id, data)
            logger.info(f"Stored in Firebase with ID: {doc_id}")
            self.mark_as_synced(user_id, session_id)
        except Exception as e:
            logger.error(f"Error storing in Firebase: {str(e)}. Data stored locally.")

    def mark_as_synced(self, user_id: str, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE sessions SET synced = 1 WHERE user_id = ? AND session_id = ?', (user_id, session_id))
        conn.commit()
        conn.close()
    
    def sync_from_firebase(self, user_id: str):
        try:
            logger.info(f"Fetching sessions for user {user_id} from Firebase")
            user_doc_ref = firebase_connection.db.collection(settings.ROOTCOLECCTION).document(user_id)
            session_collections = user_doc_ref.collections()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for session_collection in session_collections:
                for session in session_collection.stream():
                    data = session.to_dict()
                    session_id = data.get("sessionId")
                    guid = data.get('guid')
                    timestamp = data.get('timestamp', time.time())

                    # Verificar si el GUID ya existe en la base de datos local
                    cursor.execute('SELECT COUNT(*) FROM sessions WHERE guid = ?', (guid,))
                    count = cursor.fetchone()[0]

                    if count == 0:
                        # Insertar en la base de datos local si no existe
                        cursor.execute('''
                            INSERT OR IGNORE INTO sessions (user_id, session_id, guid, data, timestamp, synced) 
                            VALUES (?, ?, ?, ?, ?, 1)
                        ''', (user_id, session_id, guid, json.dumps(data), timestamp))
                    else:
                        # Actualizar la entrada existente si los GUID coinciden
                        cursor.execute('''
                            UPDATE sessions
                            SET data = ?, timestamp = ?, synced = 1
                            WHERE guid = ?
                        ''', (json.dumps(data), timestamp, guid))

            conn.commit()
            conn.close()
            logger.info(f"Synced sessions for user {user_id} from Firebase to local database")
        except Exception as e:
            logger.error(f"Error syncing sessions for user {user_id} from Firebase: {str(e)}")
            
    def sync_to_firebase(self, user_id: str):
        try:
            logger.info(f"Starting synchronization from local database to Firebase for user {user_id}")

            # Conectar a la base de datos local
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Obtener todas las sesiones locales del usuario
            cursor.execute('SELECT session_id, guid, data, timestamp FROM sessions WHERE user_id = ?', (user_id,))
            local_sessions = cursor.fetchall()

            # Conectar a Firebase y obtener sesiones del usuario
            user_doc_ref = firebase_connection.db.collection(settings.ROOTCOLECCTION).document(user_id)
            firebase_sessions = {}
            for session_collection in user_doc_ref.collections():
                for session in session_collection.stream():
                    data = session.to_dict()
                    guid = data.get('guid')
                    if guid:
                        firebase_sessions[guid] = data

            # Sincronizar sesiones locales con Firebase
            for session_id, guid, data, timestamp in local_sessions:
                if guid not in firebase_sessions:
                    # Si la sesión no está en Firebase, subirla
                    try:
                        self.store_in_firebase(user_id, session_id, json.loads(data))
                        logger.info(f"Synced local session {session_id} with GUID {guid} to Firebase")
                    except Exception as e:
                        logger.error(f"Error syncing session {session_id} with GUID {guid} to Firebase: {str(e)}")
                else:
                    logger.info(f"Session {session_id} with GUID {guid} already exists in Firebase")

            conn.close()
            logger.info(f"Completed synchronization from local database to Firebase for user {user_id}")
        except Exception as e:
            logger.error(f"Error during local to Firebase sync for user {user_id}: {str(e)}")



session_storage = SessionStorage(settings.SQLITE_DB_PATH)

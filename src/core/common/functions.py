from ...core.storage.firebase import firebase_connection
from ...core.config import settings
from typing import Dict, Any, List, Optional
from ..storage.session_storage import session_storage
import time
import datetime

class ToolFunctions():
    
    def sendToFirebase(model_name, messages, kwargs, filtered_kwargs, resp, logger):
        if "session" in kwargs and kwargs["session"] is not None and "userId" in kwargs["session"] and "sessionId" in kwargs["session"]:
            session = kwargs["session"]
            now = datetime.datetime.now()
                ## get last item from message extract the content
            llm_data = {
                    "create": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": model_name,
                    "request": messages[-1]["content"],
                    "options": filtered_kwargs,
                    "response": resp,
                    "messages": messages,
                    "timestamp": time.time()
                }
            doc_id = firebase_connection.add_document(f'{settings.ROOTCOLECCTION}/{session.get("userId")}/{session.get("sessionId")}', llm_data)
            logger.info(f"Saved LLM interaction to Firebase with ID: {doc_id}")
            
    def saveSessionData(model_name: str, input_data: Any, kwargs: Dict[str, Any], filtered_kwargs: Dict[str, Any], response, logger) -> None:
        try:
            if "session" in kwargs and kwargs["session"] is not None and "userId" in kwargs["session"] and "sessionId" in kwargs["session"]:
                session = kwargs["session"]
                session_data = {
                    "model": model_name,
                    "request": str(input_data),
                    "options": filtered_kwargs,
                    "response": response,
                    "timestamp": time.time()
                }
                session_storage.store_session_data(session["userId"], session["sessionId"], session_data)
        except Exception as e:
            logger.error(f"Error storing session data: {str(e)}")
    
    def sync_databases(logger) -> None:
        try:
            logger.info("Starting session data synchronization")
            session_storage.sync_with_firebase()
            logger.info("Session data synchronization completed")
        except Exception as e:
            logger.error(f"Error during session data synchronization: {str(e)}")
            
# Asegúrate de que la clase esté siendo exportada
__all__ = ['ToolFunctions']
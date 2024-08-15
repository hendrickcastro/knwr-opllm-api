from fastapi import APIRouter, HTTPException
from ...core.storage.session_storage import session_storage
import logging

logger = logging.getLogger(__name__)

router_sessions = APIRouter()

@router_sessions.get("/list/{user_id}/user")
async def get_all_sessions(user_id: str):
    try:
        sessions = session_storage.list_sessions(user_id)
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error retrieving sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving sessions: {str(e)}")

@router_sessions.get("/{user_id}/sessions/{session_id}/items")
async def get_items_in_session(user_id: str, session_id: str):
    try:
        items = session_storage.list_items_in_session(user_id, session_id)
        return {"items": items}
    except Exception as e:
        logger.error(f"Error retrieving items in session {session_id} for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving items in session: {str(e)}")

@router_sessions.post("/sync_from_firebase/{user_id}")
async def sync_from_firebase(user_id: str):
    try:
        session_storage.sync_from_firebase(user_id)
        return {"message": "Sync from Firebase completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing from Firebase: {str(e)}")
    

@router_sessions.post("/sync_to_firebase/{user_id}")
async def sync_to_firebase(user_id: str):
    try:
        session_storage.sync_to_firebase(user_id)
        return {"message": "Sync to Firebase completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing to Firebase: {str(e)}")
    
    
@router_sessions.delete("/{user_id}/userid/{session_id}/sessionid/{guid}/guid")
async def sync_to_firebase(user_id: str, session_id: str, guid: str):
    try:
        session_storage.delete_session_data(user_id, session_id, guid)
        return {"message": "Sync to Firebase completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing to Firebase: {str(e)}")
    
    
@router_sessions.put("/{user_id}/session/{session_id}/guid/{guid}")
async def update_session_data(user_id: str, session_id: str, guid: str, new_data: dict):
    try:
        session_storage.update_session_data(user_id, session_id, guid, new_data)
        return {"message": "Session data updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating session data: {str(e)}")

# Endpoint para añadir un ítem al contexto de una sesión
@router_sessions.post("/{user_id}/session/{session_id}/context/add_item")
async def add_item_to_session_context(user_id: str, session_id: str, item_data: dict):
    try:
        session_storage.add_item_to_session_context(user_id, session_id, item_data)
        return {"message": f"Item added to context for session {session_id}."}
    except Exception as e:
        logger.error(f"Error adding item to context for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error adding item to context: {str(e)}")

# Endpoint para obtener el contexto completo de una sesión
@router_sessions.get("/{user_id}/session/{session_id}/context")
async def get_session_context(user_id: str, session_id: str):
    try:
        context = session_storage.get_session_context(session_id)
        return {"context": context}
    except Exception as e:
        logger.error(f"Error retrieving context for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")

# Endpoint para sincronizar el contexto de una sesión con Firebase
@router_sessions.post("/{user_id}/session/{session_id}/context/sync_to_firebase")
async def sync_session_context_to_firebase(user_id: str, session_id: str):
    try:
        context = session_storage.get_session_context(session_id)
        if context:
            session_storage.save_session_context(user_id, session_id, context)
            return {"message": f"Context for session {session_id} synced to Firebase."}
        else:
            return {"message": f"No context found for session {session_id}."}
    except Exception as e:
        logger.error(f"Error syncing context for session {session_id} to Firebase: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error syncing context: {str(e)}")
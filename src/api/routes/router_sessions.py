from fastapi import APIRouter, HTTPException
from ...core.storage.session_storage import session_storage
import logging

logger = logging.getLogger(__name__)

router_sessions = APIRouter()

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
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime

class Embedding(BaseModel):
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatSession(BaseModel):
    session_id: str
    messages: List[ChatMessage]
    model_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class User(BaseModel):
    user_id: str
    username: str
    email: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
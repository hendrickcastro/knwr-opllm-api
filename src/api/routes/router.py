from fastapi import APIRouter, HTTPException
from typing import List, Dict
from ...core.agents.autoagent import auto_agent_factory
from ...models.model_manager import model_manager
from ...entity.Class import RequestBasic, ChatResponse, ChatRequest, AutoAgentRequest, AutoAgentResponse, GenerateRequest, GenerateResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/generate", response_model=ChatResponse)
async def generate_text(request: RequestBasic):
    try:
        logger.info(f"Received generate request: {request.model_dump()}")
        kwargs = request.model_dump(exclude={"modelName", "messages", "max_tokens", "temperature", "prompt"})
        
        response = model_manager.generate(
            modelName=request.modelName,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            **kwargs
        )
        return ChatResponse.from_model_response(response)
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request for model: {request.modelName}")
        kwargs = request.model_dump(exclude={"modelName", "messages", "max_tokens", "temperature"})
        
        response = model_manager.generate_chat(
            modelName=request.modelName,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            **kwargs
        )
        
        return ChatResponse.from_model_response(response)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

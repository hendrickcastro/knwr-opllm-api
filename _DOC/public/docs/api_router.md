
## Archivo: router.py
### Ruta Relativa: ../src\api\routes\router.py

```python
from fastapi import APIRouter, HTTPException
from typing import List, Dict
from ...core.agents.autoagent import auto_agent_factory
from ...models.model_manager import model_manager
from ...entity.Class import ChatResponse, ChatRequest, AutoAgentRequest, AutoAgentResponse, GenerateRequest, GenerateResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    try:
        logger.info(f"Received generate request: {request.dict()}")
        kwargs = {}

        kwargs['temperature'] = request.temperature
        generated_text = model_manager.generate(
            request.model,
            request.prompt,
            max_tokens=request.max_tokens,
            **kwargs
        )
        return GenerateResponse(generated_text=generated_text)
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request for model: {request.model_name}")
        kwargs = request.dict(exclude={"model_name", "messages", "max_tokens", "temperature"})
        
        response = model_manager.generate_chat(
            model_name=request.model_name,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            **kwargs
        )
        
        return ChatResponse.from_model_response(response)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/autoagent", response_model=AutoAgentResponse)
async def process_auto_agent(request: AutoAgentRequest):
    try:
        agent = auto_agent_factory.create_agent(request.model_name, request.task_description)
        response = agent.process_input(request.user_input)
        return AutoAgentResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

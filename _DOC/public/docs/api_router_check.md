
## Archivo: router_check.py
### Ruta Relativa: ../src\api\routes\router_check.py

```python
from fastapi import APIRouter, HTTPException
from typing import List, Dict
from ...models.model_manager import model_manager
import logging

logger = logging.getLogger(__name__)

router_check = APIRouter()

@router_check.get("/models", response_model=List[Dict[str, str]])
async def list_models():
    try:
        return model_manager.list_loaded_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_check.post("/load_model")
async def load_model(model_name: str, model_type: str):
    try:
        model_manager.load_model(model_name, model_type)
        return {"message": f"Model {model_name} loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router_check.get("/check_ollama")
async def check_ollama():
    try:
        return model_manager.check_ollama_connection()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router_check.get("/check_ollama_models")
async def check_ollama_models():
    try:
        return model_manager.list_ollama_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

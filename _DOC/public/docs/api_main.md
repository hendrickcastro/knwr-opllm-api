
## Archivo: main.py
### Ruta Relativa: ../api\main.py

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.routes import router
import uvicorn
from core.utils import setup_logger

logger = setup_logger(__name__)

app = FastAPI(debug=True)  # Enable debug mode

# Include the router without a prefix
app.include_router(router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": f"An unexpected error occurred: {str(exc)}"},
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

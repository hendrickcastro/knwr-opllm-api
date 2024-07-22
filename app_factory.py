from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.routes import router
from core.utils import setup_logger

logger = setup_logger(__name__)

def create_app() -> FastAPI:
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

    return app

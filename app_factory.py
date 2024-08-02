from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from src.api.routes import router, router_storage, router_check
from src.core.utils import setup_logger

logger = setup_logger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(debug=True)  # Enable debug mode

    # Include the main router without a prefix
    app.include_router(router)

    # Include the new routers with their respective prefixes
    app.include_router(router_storage, prefix="/storage", tags=["storage"])
    app.include_router(router_check, prefix="/check", tags=["check"])

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": f"An unexpected error occurred: {str(exc)}"},
        )

    return app
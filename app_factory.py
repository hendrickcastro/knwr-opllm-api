from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Importa el middleware de CORS
from src.api.routes import router, router_storage, router_sessions, router_check
from src.core.utils import setup_logger

logger = setup_logger(__name__)

def create_app() -> FastAPI:
    app = FastAPI(debug=True)  # Enable debug mode

    # Configura CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:4321", "http://localhost:3000"],  # Especifica el origen permitido
        allow_credentials=True,
        allow_methods=["*"],  # Permite todos los métodos HTTP (GET, POST, etc.)
        allow_headers=["*"],  # Permite todas las cabeceras
    )

    # Incluye el enrutador principal sin prefijo
    app.include_router(router)

    # Incluye los nuevos enrutadores con sus respectivos prefijos
    app.include_router(router_sessions, prefix="/sessions", tags=["sessions"])
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

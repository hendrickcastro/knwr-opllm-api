import ptvsd
import uvicorn
from app_factory import create_app

app = create_app()

if __name__ == "__main__":
    # Habilitar ptvsd (Python Tools for Visual Studio Debugging)
    ptvsd.enable_attach(address=('0.0.0.0', 5679), redirect_output=True)

    # No detener la ejecución esperando el depurador para permitir recarga automática
    # ptvsd.wait_for_attach()

    # Pasar la aplicación como cadena de importación para soportar recarga automática
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

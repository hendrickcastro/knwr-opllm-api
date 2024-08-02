
## Archivo: main.py
### Ruta Relativa: ../main.py

```python
import debugpy
import uvicorn
from app_factory import create_app

app = create_app()

if __name__ == "__main__":
    # Habilitar debugpy (Python Tools for Visual Studio Debugging)
    debugpy.listen(("0.0.0.0", 5679))
    print("Esperando al depurador para adjuntar...")
    # Puedes descomentar la siguiente línea si deseas que la ejecución se detenga hasta que el depurador se conecte
    # debugpy.wait_for_client()

    # Pasar la aplicación como cadena de importación para soportar recarga automática
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

```

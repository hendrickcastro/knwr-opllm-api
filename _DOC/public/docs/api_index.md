
## Archivo: index.py
### Ruta Relativa: ../src\tools\langchain\index.py

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Obtener las variables de entorno
api_key = os.getenv("OPENAI_API_KEY")
endpoint = os.getenv("LANGCHAIN_ENDPOINT")
tracing = os.getenv("LANGCHAIN_TRACING_V2")
project = os.getenv("LANGCHAIN_PROJECT")

# Verificar que las variables de entorno se están cargando correctamente
print(f"API Key: {api_key}")
print(f"Endpoint: {endpoint}")
print(f"Tracing: {tracing}")
print(f"Project: {project}")

# Inicializar el modelo de lenguaje con la clave API
llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

# Invocar el modelo con un prompt
try:
    response = llm.invoke("Hello, world!")
    print(response)
except Exception as e:
    print(f"Error: {e}")

```


## Archivo: testGrok.py
### Ruta Relativa: ../src\tools\testGrok.py

```python
import os
from groq import Groq

# Asegúrate de tener tu API key configurada en las variables de entorno
os.environ["GROQ_API_KEY"] = "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"

# Inicializar el cliente de Groq con tu API key
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Crear una solicitud de chat para el modelo llama3-groq-70b-8192-tool-use-preview
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Escribe una función en Python que tome una lista de números avansado y devuelva una nueva lista que contenga solo los números pares de la lista original. La función debe llamarse 'filtrar_numeros_pares'.",
        }
    ],
    # model="llama-3.1-70b-versatile",
    # model="llama-3.1-8b-instant",
    # model="llama3-groq-70b-8192-tool-use-preview",
    model="llama3.1:8b-instruct-fp16"
    temperature=0
)

# Imprimir el contenido de la respuesta
print(chat_completion.choices[0].message.content)

```

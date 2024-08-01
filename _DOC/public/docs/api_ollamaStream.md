
## Archivo: ollamaStream.py
### Ruta Relativa: ../src\tools\ollamaStream.py

```python
import asyncio
from ollama import AsyncClient

async def chat():
  message = {'role': 'user', 'content': 'Why is the sky blue?'}
  response = await AsyncClient().chat(model='llama3.1', messages=[message])

asyncio.run(chat())
```

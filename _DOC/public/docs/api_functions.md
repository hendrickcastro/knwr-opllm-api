
## Archivo: functions.py
### Ruta Relativa: ../src\core\common\functions.py

```python
from ...core.storage.firebase import firebase_connection
from ...core.config import settings
import time
import datetime

class ToolFunctions():
    
    def sendToFirebase(model_name, messages, kwargs, filtered_kwargs, resp, logger):
        if "session" in kwargs and kwargs["session"] is not None and "userId" in kwargs["session"] and "sessionId" in kwargs["session"]:
            session = kwargs["session"]
            now = datetime.datetime.now()
                ## get last item from message extract the content
            llm_data = {
                    "create": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": model_name,
                    "request": messages[-1]["content"],
                    "options": filtered_kwargs,
                    "response": resp,
                    "messages": messages,
                    "timestamp": time.time()
                }
            doc_id = firebase_connection.add_document(f'{settings.ROOTCOLECCTION}/{session.get("userId")}/{session.get("sessionId")}', llm_data)
            logger.info(f"Saved LLM interaction to Firebase with ID: {doc_id}")
            
# Asegúrate de que la clase esté siendo exportada
__all__ = ['ToolFunctions']
```


## Archivo: chunk_handler.py
### Ruta Relativa: ../api\chunks\chunk_handler.py

```python
from typing import List, Union
from .text_chunks import text_chunker
from .code_chunks import code_chunker
from core.utils import setup_logger

logger = setup_logger(__name__)

class ChunkHandler:
    def __init__(self):
        self.text_chunker = text_chunker
        self.code_chunker = code_chunker

    def process_chunks(self, content: str, content_type: str) -> List[str]:
        try:
            if content_type == 'text':
                return self.text_chunker.chunk_text(content)
            elif content_type == 'code':
                return self.code_chunker.chunk_code(content)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")
        except Exception as e:
            logger.error(f"Error processing chunks: {str(e)}")
            raise

    def set_text_chunk_size(self, size: int) -> None:
        self.text_chunker.set_chunk_size(size)

    def set_text_overlap(self, overlap: int) -> None:
        self.text_chunker.set_overlap(overlap)

    def set_code_max_lines(self, max_lines: int) -> None:
        self.code_chunker.set_max_lines(max_lines)

chunk_handler = ChunkHandler()
```


## Archivo: text_chunks.py
### Ruta Relativa: ../api\chunks\text_chunks.py

```python
from typing import List
from core.utils import setup_logger

logger = setup_logger(__name__)

class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[str]:
        try:
            words = text.split()
            chunks = []
            start = 0
            
            while start < len(words):
                end = start + self.chunk_size
                chunk = ' '.join(words[start:end])
                chunks.append(chunk)
                start = end - self.overlap

            logger.info(f"Text chunked into {len(chunks)} parts")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise

    def set_chunk_size(self, size: int) -> None:
        self.chunk_size = size
        logger.info(f"Chunk size set to {size}")

    def set_overlap(self, overlap: int) -> None:
        self.overlap = overlap
        logger.info(f"Overlap set to {overlap}")

text_chunker = TextChunker()
```

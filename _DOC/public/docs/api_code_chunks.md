
## Archivo: code_chunks.py
### Ruta Relativa: ../api\chunks\code_chunks.py

```python
import re
from typing import List
from core.utils import setup_logger

logger = setup_logger(__name__)

class CodeChunker:
    def __init__(self, max_lines: int = 50):
        self.max_lines = max_lines

    def chunk_code(self, code: str, language: str) -> List[str]:
        try:
            lines = code.split('\n')
            chunks = []
            current_chunk = []

            for line in lines:
                current_chunk.append(line)
                if len(current_chunk) >= self.max_lines or self._is_chunk_boundary(line, language):
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []

            if current_chunk:
                chunks.append('\n'.join(current_chunk))

            logger.info(f"Code chunked into {len(chunks)} parts")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking code: {str(e)}")
            raise

    def _is_chunk_boundary(self, line: str, language: str) -> bool:
        patterns = {
            "python": [
                r'^\s*def\s+\w+\s*\(.*\):',  # Function definition
                r'^\s*class\s+\w+.*:',       # Class definition
                r'^\s*if\s+__name__\s*==\s*[\'"]__main__[\'"]:'  # Main block
            ],
            "javascript": [
                r'^\s*function\s+\w+\s*\(.*\)\s*{',  # Function definition
                r'^\s*class\s+\w+\s*{',              # Class definition
                r'^\s*const\s+\w+\s*=\s*function\s*\(.*\)\s*{',  # Arrow function
                r'^\s*export\s+',                    # Export statement
                r'^\s*import\s+'                     # Import statement
            ],
            "csharp": [
                r'^\s*public\s+(class|interface|struct|enum)\s+\w+',  # Class, interface, struct, or enum
                r'^\s*(public|private|protected)\s+\w+\s+\w+\s*\(.*\)',  # Method definition
                r'^\s*namespace\s+',                 # Namespace
                r'^\s*using\s+'                      # Using statement
            ]
        }
        
        return any(re.match(pattern, line) for pattern in patterns.get(language, []))

    def set_max_lines(self, max_lines: int) -> None:
        self.max_lines = max_lines
        logger.info(f"Max lines per chunk set to {max_lines}")

code_chunker = CodeChunker()
```

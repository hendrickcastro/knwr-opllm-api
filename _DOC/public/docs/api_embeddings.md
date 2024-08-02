
## Archivo: embeddings.py
### Ruta Relativa: ../src\models\embeddings.py

```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
from ..core.utils import setup_logger

logger = setup_logger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> List[float]:
        try:
            embedding = self.model.encode([text])[0]
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def compare_embeddings(self, embedding1: List[float], embedding2: List[float]) -> float:
        try:
            return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        except Exception as e:
            logger.error(f"Error comparing embeddings: {str(e)}")
            raise

embedding_generator = EmbeddingGenerator()
```

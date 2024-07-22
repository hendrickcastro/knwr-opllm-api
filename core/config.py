import os
from dotenv import load_dotenv
from typing import Dict, Any

load_dotenv()

class Settings:
    MONGODB_URI: str = os.getenv("MONGODB_URI")
    MONGODB_USER: str = os.getenv("MONGODB_USER")
    MONGODB_PASS: str = os.getenv("MONGODB_PASS")
    MONGODB_DB: str = os.getenv("MONGODB_DB", "cashabotorllm")
    
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Configuración para futuros modelos
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY")
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY")
    GROK_API_KEY: str = os.getenv("GROK_API_KEY")

    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "ollama": {
            "base_url": OLLAMA_BASE_URL,
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "api_key": OPENAI_API_KEY,
        },
        "anthropic": {
            "base_url": "https://api.anthropic.com",
            "api_k,ey": ANTHROPIC_API_KEY,
        },
        "huggingface": {
            "base_url": "https://api.huggingface.co",
            "api_key": HUGGINGFACE_API_KEY,
        },
        "grok": {
            "base_url": "https://api.grok.com",
            "api_key": GROK_API_KEY
        },
    }

    @property
    def DATABASE_URL(self):
        return f"mongodb://{self.MONGODB_USER}:{self.MONGODB_PASS}@{self.MONGODB_URI.split('://')[1]}"

settings = Settings()
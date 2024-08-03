import os
from dotenv import load_dotenv
from typing import Dict, Any
from ..core.storage.firebase import firebase_connection

load_dotenv()

class Settings:
    def __init__(self):
        self.firebase = firebase_connection

    def _get_config(self, key: str, default: Any = None) -> Any:
        value = self.firebase.get(key)
        if value is None:
            value = os.getenv(key, default)
        return value

    @property
    def MONGODB_URI(self) -> str:
        return self._get_config("MONGODB_URI")

    @property
    def MONGODB_USER(self) -> str:
        return self._get_config("MONGODB_USER")

    @property
    def MONGODB_PASS(self) -> str:
        return self._get_config("MONGODB_PASS")

    @property
    def MONGODB_DB(self) -> str:
        return self._get_config("MONGODB_DB", "cashabotorllm")

    @property
    def OLLAMA_BASE_URL(self) -> str:
        return self._get_config("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def OPENAI_API_KEY(self) -> str:
        return self._get_config("OPENAI_API_KEY")

    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return self._get_config("ANTHROPIC_API_KEY")

    @property
    def HUGGINGFACE_API_KEY(self) -> str:
        return self._get_config("HUGGINGFACE_API_KEY")

    @property
    def GROQ_API_KEY(self) -> str:
        return self._get_config("GROQ_API_KEY")
    
    @property
    def DEBUGG(self) -> str:
        return self._get_config("ENV") == "development"
    
    @property
    def ROOTCOLECCTION(self) -> str:
        return self._get_config("ROOTCOLECCTION")

    @property
    def DEFAULT_MODELS(self) -> Dict[str, str]:
        default = {
            "gpt-4o-mini": "openai",
            "gpt-4o-coder": "grok",
            "mistralai/Mixtral-8x7B-Instruct-v0.1": "huggingface",
            "claude-3-5-sonnet-20240620": "anthropic",
            "llama-3.1-70b-versatile": "groq",
            "llama-3.1-405b-reasoning": "groq",
            "llama3-groq-70b-8192-tool-use-preview": "groq"
        }
        return self._get_config("DEFAULT_MODELS", default)

    @property
    def DATABASE_URL(self):
        return f"mongodb://{self.MONGODB_USER}:{self.MONGODB_PASS}@{self.MONGODB_URI.split('://')[1]}"

    @property
    def SQLITE_DB_PATH(self) -> str:
        return self._get_config("SQLITE_DB_PATH", "./db/local_sessions.db")

settings = Settings()

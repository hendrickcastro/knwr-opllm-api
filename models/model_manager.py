from typing import Dict, Any, List, Optional
from .base_model import BaseModel
from .ollama_model import OllamaModel
from .huggingface_model import HuggingFaceModel
from .openai_model import OpenAIModel
from .anthropic_model import AnthropicModel
from .grok_model import GrokModel
from core.utils import setup_logger
import requests
from core.config import settings

logger = setup_logger(__name__)

class ModelManager:
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.default_models = {
            "gpt-4o-mini": "openai",
            "gpt-4o-coder": "grok",
            "microsoft/codebert-base": "huggingface",
            "claude-3-5-sonnet-20240620": "anthropic"
        }
        self._load_ollama_models()
        logger.info("Initializing ModelManager")

    def _load_ollama_models(self) -> None:
        try:
            response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                ollama_model_names = [model["name"] for model in models]
                for model_name in ollama_model_names:
                    self.default_models[model_name] = "ollama"
                logger.info(f"Ollama models loaded: {ollama_model_names}")
            else:
                logger.error(f"Failed to get Ollama models. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")

    def _load_model(self, model_name: str, model_type: str) -> None:
        if model_name in self.models:
            logger.info(f"Model {model_name} already loaded")
            return

        model_classes = {
            "ollama": OllamaModel,
            "huggingface": HuggingFaceModel,
            "openai": OpenAIModel,
            "anthropic": AnthropicModel,
            "grok": GrokModel
        }

        if model_type not in model_classes:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info(f"Creating instance of {model_type} model: {model_name}")
        model = model_classes[model_type](model_name)
        logger.info(f"Loading model: {model_name}")
        model.load()
        self.models[model_name] = model
        logger.info(f"Model {model_name} of type {model_type} loaded successfully")

    def load_model(self, model_name: str, model_type: str) -> None:
        self._load_model(model_name, model_type)

    def get_model(self, model_name: str, model_type: Optional[str] = None) -> Optional[BaseModel]:
        if model_name not in self.models:
            if model_type is None:
                model_type = self.default_models.get(model_name)
                if model_type is None:
                    logger.error(f"Model type for {model_name} not provided and not found in default models")
                    return None
            try:
                self.load_model(model_name, model_type)
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {str(e)}")
                return None
        return self.models.get(model_name)

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        return [model.get_info() for model in self.models.values()]

    def generate(self, model_name: str, prompt: str, max_tokens: Optional[int] = None, model_type: Optional[str] = None, **kwargs) -> str:
        model = self.get_model(model_name, model_type)
        if model is None:
            return f"Model {model_name} not loaded"
        return model.generate(prompt, max_tokens, **kwargs)

model_manager = ModelManager()

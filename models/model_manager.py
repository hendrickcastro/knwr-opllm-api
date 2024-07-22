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
        logger.info("Initializing ModelManager")
        self._load_default_models()

    def _load_default_models(self):
        default_models = [
            ("gpt-4o-mini", "openai"),
            ("gpt-4o-coder", "grok"),
            ("microsoft/codebert-base", "huggingface"),
            ("claude-3-5-sonnet-20240620", "anthropic")
        ]
        logger.info(f"Attempting to load default models: {default_models}")
        for model_name, model_type in default_models:
            try:
                logger.info(f"Attempting to load model: {model_name} of type {model_type}")
                self.load_model(model_name, model_type)
            except Exception as e:
                logger.warning(f"Failed to load default model {model_name}: {str(e)}", exc_info=True)
        
        self._load_ollama_models()

    def _load_ollama_models(self):
        try:
            ollama_models = self._get_ollama_models()
            for model_name in ollama_models:
                try:
                    logger.info(f"Attempting to load Ollama model: {model_name}")
                    self.load_model(model_name, "ollama")
                except Exception as e:
                    logger.warning(f"Failed to load Ollama model {model_name}: {str(e)}", exc_info=True)
        except Exception as e:
            logger.error(f"Failed to load Ollama models: {str(e)}", exc_info=True)

    def _get_ollama_models(self) -> List[str]:
        try:
            response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            else:
                logger.error(f"Failed to get Ollama models. Status code: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")
            return []

    def load_model(self, model_name: str, model_type: str) -> None:
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

    def get_model(self, model_name: str) -> BaseModel:
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        return self.models[model_name]

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        return [model.get_info() for model in self.models.values()]

    def generate(self, model_name: str, prompt: str, max_tokens: Optional[int] = None, **kwargs) -> str:
        model = self.get_model(model_name)
        return model.generate(prompt, max_tokens, **kwargs)

model_manager = ModelManager()
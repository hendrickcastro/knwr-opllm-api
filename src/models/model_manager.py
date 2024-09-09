import requests
from typing import Dict, Any, List, Optional
from ..contract.IClient import IClient
from .client.ollama import OllamaModel
from .client.huggingface import HuggingFaceModel
from .client.openai import OpenAIModel
from .client.anthropic import AnthropicModel
from .client.groq import GroqModel
from ..core.utils import setup_logger
from ..core.config import settings
from ..core.common.functions import ToolFunctions

logger = setup_logger(__name__)

class ModelManager:
    def __init__(self):
        self.models: Dict[str, IClient] = {}
        self.default_models = settings.DEFAULT_MODELS
        self.tool_functions = ToolFunctions()
        self._load_ollama_models()
        logger.info("Initializing ModelManager")
        self.tool_functions.sync_databases(logger)
        
    def _filter_kwargs_for_model(self, model: IClient, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if hasattr(model, '_filter_kwargs'):
            return model._filter_kwargs(**kwargs)
        return kwargs

    def _load_ollama_models(self) -> None:
        try:
            response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                ollama_modelNames = [model["name"] for model in models]
                for modelName in ollama_modelNames:
                    self.default_models[modelName] = "ollama"
                    self.default_models[f"{modelName}:latest"] = "ollama"
                logger.info(f"Ollama models loaded: {ollama_modelNames}")
            else:
                logger.error(f"Failed to get Ollama models. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")

    def _load_model(self, modelName: str, model_type: str) -> None:
        if modelName in self.models:
            logger.info(f"Model {modelName} already loaded")
            return

        model_classes = {
            "ollama": OllamaModel,
            "huggingface": HuggingFaceModel,
            "openai": OpenAIModel,
            "anthropic": AnthropicModel,
            "groq": GroqModel
        }

        if model_type not in model_classes:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        try:
            logger.info(f"Creating instance of {model_type} model: {modelName}")
            model = model_classes[model_type](modelName)
            logger.info(f"Loading model: {modelName}")
            model.load()
            self.models[modelName] = model
            logger.info(f"Model {modelName} of type {model_type} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model {modelName}: {str(e)}")
            raise

    def load_model(self, modelName: str, model_type: str) -> None:
        self._load_model(modelName, model_type)

    def get_model(self, modelName: str, model_type: Optional[str] = None) -> Optional[IClient]:
        if modelName not in self.models:
            if model_type is None:
                model_type = self.default_models.get(modelName)
                if model_type is None:
                    logger.error(f"Model type for {modelName} not provided and not found in default models")
                    return None
            try:
                self.load_model(modelName, model_type)
            except Exception as e:
                logger.error(f"Failed to load model {modelName}: {str(e)}")
                return None
        return self.models.get(modelName)

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        return [model.get_info() for model in self.models.values()]

    def _generate_response(self, modelName: str, input_data: Any, max_tokens: Optional[int], temperature: float, model_type: Optional[str], method: str, **kwargs) -> Optional[Any]:
        try:
            model = self.get_model(modelName, model_type)
            if model is None:
                logger.error(f"Model {modelName} not loaded")
                return f"Model {modelName} not loaded"
            
            filtered_kwargs = self._filter_kwargs_for_model(model, kwargs)
            
            if (settings.DEBUGG):
                for key, value in filtered_kwargs.items():
                    print(f"{key}: {value}")

            if method == "generate":
                response = model.generate(input_data, max_tokens, temperature, **filtered_kwargs)
            elif method == "chat":
                response = model.generate_chat(input_data, max_tokens, temperature, **filtered_kwargs)
            else:
                logger.error(f"Unsupported method: {method}")
                return f"Unsupported method: {method}"
            
            self.tool_functions.saveSessionData(modelName, input_data, kwargs, filtered_kwargs, response, logger)
            
            return response
        except Exception as e:
            raise e

    def generate(self, modelName: str, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7, model_type: Optional[str] = None, **kwargs) -> Any:
        return self._generate_response(modelName, prompt, max_tokens, temperature, model_type, method="generate", **kwargs)

    def generate_chat(self, modelName: str, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, model_type: Optional[str] = None, **kwargs) -> Optional[object]:
        messages_dict = [msg.dict() for msg in messages]
        try:
            resp = self._generate_response(modelName, messages_dict, max_tokens, temperature, model_type, method="chat", **kwargs)
            return resp
        except Exception as e:
            raise e

model_manager = ModelManager()
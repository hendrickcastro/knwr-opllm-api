import requests
from typing import Dict, Any, List, Optional
from .base_model import BaseModel
from .client.ollama import OllamaModel
from .client.huggingface import HuggingFaceModel
from .client.openai import OpenAIModel
from .client.anthropic import AnthropicModel
from .client.grok import GrokModel
from ..core.utils import setup_logger
from ..core.config import settings
from .prompt_factory import PromptHandlerFactory

logger = setup_logger(__name__)

class ModelManager:
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.default_models = settings.DEFAULT_MODELS
        self._load_ollama_models()
        logger.info("Initializing ModelManager")

    def _load_ollama_models(self) -> None:
        """
        Load available Ollama models from the Ollama API and update the default models dictionary.
        """
        try:
            response = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                ollama_model_names = [model["name"] for model in models]
                for model_name in ollama_model_names:
                    self.default_models[model_name] = "ollama"
                    self.default_models[f"{model_name}:latest"] = "ollama"
                logger.info(f"Ollama models loaded: {ollama_model_names}")
            else:
                logger.error(f"Failed to get Ollama models. Status code: {response.status_code}")
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {str(e)}")

    def _load_model(self, model_name: str, model_type: str) -> None:
        """
        Load a specific model based on its name and type.

        :param model_name: Name of the model to load.
        :param model_type: Type of the model (e.g., 'ollama', 'huggingface').
        """
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
        """
        Public method to load a model.

        :param model_name: Name of the model to load.
        :param model_type: Type of the model (e.g., 'ollama', 'huggingface').
        """
        self._load_model(model_name, model_type)

    def get_model(self, model_name: str, model_type: Optional[str] = None) -> Optional[BaseModel]:
        """
        Retrieve a model instance by name and type.

        :param model_name: Name of the model to retrieve.
        :param model_type: Optional type of the model if not using the default.
        :return: The model instance or None if the model cannot be loaded.
        """
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
        """
        List information about all loaded models.

        :return: A list of dictionaries containing information about each loaded model.
        """
        return [model.get_info() for model in self.models.values()]

    def _generate_response(self, model_name: str, input_data: Any, handler_method: str, max_tokens: Optional[int], temperature: float, model_type: Optional[str], **kwargs) -> Optional[str]:
        """
        Generalized method to generate a response using a specified model.

        :param model_name: Name of the model to use.
        :param input_data: The input data to format (prompt or messages).
        :param handler_method: Method of the handler to format the input data.
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: Sampling temperature for generation.
        :param model_type: Optional type of the model if not using the default.
        :param kwargs: Additional keyword arguments for model generation.
        :return: The generated response as a string, or an error message if the model is not loaded.
        """
        try:
            model = self.get_model(model_name, model_type)
            if model is None:
                logger.error(f"Model {model_name} not loaded")
                return f"Model {model_name} not loaded"

            handler = PromptHandlerFactory.get_handler(model.get_info()['type'])
            formatted_input = getattr(handler, handler_method)(input_data)

            return model.generate_chat(formatted_input, max_tokens, temperature, **kwargs)
        except Exception as e:
            logger.error(f"Error generating response with model {model_name}: {str(e)}")
            return f"Error generating response: {str(e)}"

    def generate(self, model_name: str, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7, model_type: Optional[str] = None, **kwargs) -> str:
        """
        Generates a completion based on the provided prompt using the specified model.

        :param model_name: Name of the model to use.
        :param prompt: The prompt string to generate a completion for.
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: Sampling temperature for generation.
        :param model_type: Optional type of the model if not using the default.
        :param kwargs: Additional keyword arguments for model generation.
        :return: The generated completion as a string, or an error message if the model is not loaded.
        """
        return self._generate_response(model_name, prompt, 'format_completion', max_tokens, temperature, model_type, **kwargs)

    def generate_chat(self, model_name: str, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, model_type: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Generates a chat response using the specified model.

        :param model_name: Name of the model to use.
        :param messages: List of messages as dictionaries with 'role' and 'content' keys.
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: Sampling temperature for generation.
        :param model_type: Optional type of the model if not using the default.
        :param kwargs: Additional keyword arguments for model generation.
        :return: The generated chat response as a string, or None if an error occurs.
        """
        messages_dict = [msg.dict() for msg in messages]
        return self._generate_response(model_name, messages_dict, 'format_prompt', max_tokens, temperature, model_type, **kwargs)


model_manager = ModelManager()

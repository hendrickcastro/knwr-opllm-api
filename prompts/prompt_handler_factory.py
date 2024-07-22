from .openai_prompt_handler import OpenAIPromptHandler
from .ollama_prompt_handler import OllamaPromptHandler
from .anthropic_prompt_handler import AnthropicPromptHandler
from .grok_prompt_handler import GrokPromptHandler
from .huggingface_prompt_handler import HuggingFacePromptHandler

class PromptHandlerFactory:
    @staticmethod
    def get_handler(model_type: str):
        handlers = {
            "openai": OpenAIPromptHandler(),
            "ollama": OllamaPromptHandler(),
            "anthropic": AnthropicPromptHandler(),
            "grok": GrokPromptHandler(),
            "huggingface": HuggingFacePromptHandler()
        }
        return handlers.get(model_type.lower())

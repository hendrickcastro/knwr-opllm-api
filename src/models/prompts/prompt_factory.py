from .openai_prompt import OpenAIPromptHandler
from .ollama_prompt import OllamaPromptHandler
from .anthropic_prompt import AnthropicPromptHandler
from .grok_prompt import GrokPromptHandler
from .huggingface_prompt import HuggingFacePromptHandler

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

from .prompts.openai_prompt import OpenAIPromptHandler
from .prompts.ollama_prompt import OllamaPromptHandler
from .prompts.anthropic_prompt import AnthropicPromptHandler
from .prompts.grok_prompt import GrokPromptHandler
from .prompts.huggingface_prompt import HuggingFacePromptHandler

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

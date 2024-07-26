from typing import Dict, Any
from core.utils import setup_logger, sanitize_input

logger = setup_logger(__name__)

class PromptProcessor:
    def __init__(self):
        self.prompt_templates: Dict[str, str] = {
            "chat": "Human: {input}\nAI:",
            "completion": "{input}",
            "question_answering": "Context: {context}\nQuestion: {question}\nAnswer:",
        }

    def create_prompt(self, prompt_type: str, **kwargs) -> str:
        if prompt_type not in self.prompt_templates:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        try:
            sanitized_kwargs = {k: sanitize_input(v) for k, v in kwargs.items()}
            return self.prompt_templates[prompt_type].format(**sanitized_kwargs)
        except KeyError as e:
            logger.error(f"Missing required argument for prompt type {prompt_type}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error creating prompt: {str(e)}")
            raise

    def add_prompt_template(self, name: str, template: str) -> None:
        self.prompt_templates[name] = template
        logger.info(f"Added new prompt template: {name}")

prompt_processor = PromptProcessor()
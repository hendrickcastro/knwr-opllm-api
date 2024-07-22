from typing import Any, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base_model import BaseModel
from core.utils import setup_logger
from core.config import settings

logger = setup_logger(__name__)

class HuggingFaceModel(BaseModel):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = settings.HUGGINGFACE_API_KEY
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception as e:
            logger.error(f"Error loading HuggingFace model {self.model_name}: {str(e)}")
            raise

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            max_length = inputs["input_ids"].shape[1] + (max_tokens or 100)
            outputs = self.model.generate(**inputs, max_length=max_length)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating with HuggingFace model {self.model_name}: {str(e)}")
            raise

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "huggingface"}
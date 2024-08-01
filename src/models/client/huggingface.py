from typing import Any, Dict, Optional, List
from ...contract.IClient import IClient
from ...core.config import settings
from ...core.utils import setup_logger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch

logger = setup_logger(__name__)

class HuggingFaceModel(IClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.hf_token = settings.HUGGINGFACE_API_KEY  # Asume que has añadido este token a tu configuración

    def load(self) -> None:
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, token=self.hf_token)
            logger.info(f"Successfully loaded model and tokenizer for {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            raise

    def generate_chat(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None, temperature: float = 0.7, **kwargs) -> Optional[object]:
        if not self.model or not self.tokenizer:
            self.load()

        try:
            # Concatenate messages into a single string
            prompt = " ".join([f"{msg['role']}: {msg['content']}" for msg in messages])
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            filter_kwargs = self._filter_kwargs(kwargs)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                **filter_kwargs
            )
            
            response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            response_dict = {
                "message": {"content": response_text},
                "done_reason": "stop",
                "done": True,
                "total_duration": len(outputs[0]),
                "prompt_eval_count": len(inputs.input_ids[0]),
                "eval_count": len(outputs[0]) - len(inputs.input_ids[0]),
            }
            
            return response_dict
        except Exception as e:
            logger.error(f"Error generating chat with HuggingFace model {self.model_name}: {str(e)}")
            raise
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: float = 0.7) -> str:
        if not self.model or not self.tokenizer:
            self.load()

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            logger.error(f"Error generating text with HuggingFace model {self.model_name}: {str(e)}")
            raise
        
    def _filter_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        accepted_params = [
            'max_length', 'min_length', 'do_sample', 'early_stopping', 'num_beams',
            'temperature', 'top_k', 'top_p', 'repetition_penalty', 'bad_words_ids',
            'bos_token_id', 'pad_token_id', 'eos_token_id', 'length_penalty',
            'no_repeat_ngram_size', 'encoder_no_repeat_ngram_size', 'num_return_sequences',
            'max_time', 'max_new_tokens', 'decoder_start_token_id', 'use_cache',
            'num_beam_groups', 'diversity_penalty', 'prefix_allowed_tokens_fn',
            'output_attentions', 'output_hidden_states', 'output_scores', 'return_dict_in_generate',
            'forced_bos_token_id', 'forced_eos_token_id', 'remove_invalid_values',
            'exponential_decay_length_penalty', 'suppress_tokens', 'begin_suppress_tokens',
            'forced_decoder_ids', 'sequence_bias', 'guidance_scale', 'low_memory'
        ]
        return {k: v for k, v in kwargs.items() if k in accepted_params}

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.model_name, "type": "huggingface"}

    def generate_embedding(self, text: str) -> List[float]:
        if not self.model or not self.tokenizer:
            self.load()

        try:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
        except Exception as e:
            logger.error(f"Error generating embedding with HuggingFace model {self.model_name}: {str(e)}")
            raise

    # Los demás métodos permanecen sin cambios

    def create_chunks(self, content: str, content_type: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def process_auto_agent(self, user_input: str, task_description: str) -> str:
        # Placeholder implementation as details are needed
        return "Not implemented"

    def get_models(self) -> List[Dict[str, str]]:
        # Hugging Face has a vast number of models, so we'll just return the current model
        return [{"id": self.model_name, "object": "model"}]

    def generate_prompt(self, prompt: str) -> str:
        # Placeholder implementation as details are needed
        return prompt

    def generate_prompts(self, messages: List[Dict[str, str]]) -> str:
        # Placeholder implementation as details are needed
        return " ".join([message["content"] for message in messages])
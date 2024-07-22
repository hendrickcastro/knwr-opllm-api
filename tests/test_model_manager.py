import pytest
from models.model_manager import ModelManager
from models.ollama_model import OllamaModel
from models.huggingface_model import HuggingFaceModel
from models.openai_model import OpenAIModel
from models.anthropic_model import AnthropicModel

def test_model_manager_generate(mock_model_manager):
    result = mock_model_manager.generate("test_ollama_model", "Test prompt")
    assert result == "Ollama generated text"

    result = mock_model_manager.generate("test_huggingface_model", "Test prompt")
    assert result == "HuggingFace generated text"

    result = mock_model_manager.generate("test_openai_model", "Test prompt")
    assert result == "OpenAI generated text"

    result = mock_model_manager.generate("test_anthropic_model", "Test prompt")
    assert result == "Anthropic generated text"

def test_model_manager_list_loaded_models(mock_model_manager):
    loaded_models = mock_model_manager.list_loaded_models()
    assert len(loaded_models) == 4
    assert {"name": "test_ollama_model", "type": "ollama"} in loaded_models
    assert {"name": "test_huggingface_model", "type": "huggingface"} in loaded_models
    assert {"name": "test_openai_model", "type": "openai"} in loaded_models
    assert {"name": "test_anthropic_model", "type": "anthropic"} in loaded_models

def test_model_manager_get_model(mock_model_manager):
    model = mock_model_manager.get_model("test_ollama_model")
    assert isinstance(model, OllamaModel)

    model = mock_model_manager.get_model("test_huggingface_model")
    assert isinstance(model, HuggingFaceModel)

    model = mock_model_manager.get_model("test_openai_model")
    assert isinstance(model, OpenAIModel)

    model = mock_model_manager.get_model("test_anthropic_model")
    assert isinstance(model, AnthropicModel)

    with pytest.raises(ValueError):
        mock_model_manager.get_model("non_existent_model")
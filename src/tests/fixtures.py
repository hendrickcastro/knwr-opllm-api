import pytest
from unittest.mock import MagicMock
from models.ollama_model import OllamaModel
from models.huggingface_model import HuggingFaceModel
from models.openai_model import OpenAIModel
from models.anthropic_model import AnthropicModel

@pytest.fixture
def mock_ollama_model():
    model = OllamaModel("test_ollama_model")
    model._is_model_available = MagicMock(return_value=True)
    model.generate = MagicMock(return_value="Ollama generated text")
    return model

@pytest.fixture
def mock_huggingface_model():
    model = HuggingFaceModel("test_huggingface_model")
    model.load = MagicMock()
    model.generate = MagicMock(return_value="HuggingFace generated text")
    return model

@pytest.fixture
def mock_openai_model():
    model = OpenAIModel("test_openai_model")
    model.generate = MagicMock(return_value="OpenAI generated text")
    return model

@pytest.fixture
def mock_anthropic_model():
    model = AnthropicModel("test_anthropic_model")
    model.generate = MagicMock(return_value="Anthropic generated text")
    return model

@pytest.fixture
def mock_model_manager(mock_ollama_model, mock_huggingface_model, mock_openai_model, mock_anthropic_model):
    from models.model_manager import ModelManager
    manager = ModelManager()
    manager.models = {
        "test_ollama_model": mock_ollama_model,
        "test_huggingface_model": mock_huggingface_model,
        "test_openai_model": mock_openai_model,
        "test_anthropic_model": mock_anthropic_model
    }
    return manager
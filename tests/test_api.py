from fastapi.testclient import TestClient
from api.main import app
import pytest
from unittest.mock import patch

client = TestClient(app)

@pytest.fixture
def mock_model_manager():
    with patch("api.main.model_manager") as mock:
        mock.generate.return_value = "Generated text"
        mock.list_loaded_models.return_value = [
            {"name": "test_model", "type": "test_type"}
        ]
        yield mock

def test_generate_text(mock_model_manager):
    response = client.post("/generate", json={
        "model_name": "test_model",
        "prompt": "Test prompt",
        "max_tokens": 50
    })
    assert response.status_code == 200
    assert response.json() == {"generated_text": "Generated text"}
    mock_model_manager.generate.assert_called_once_with("test_model", "Test prompt", 50)

def test_generate_text_error(mock_model_manager):
    mock_model_manager.generate.side_effect = ValueError("Model not found")
    response = client.post("/generate", json={
        "model_name": "non_existent_model",
        "prompt": "Test prompt"
    })
    assert response.status_code == 400
    assert "Model not found" in response.json()["detail"]

def test_list_models(mock_model_manager):
    response = client.get("/models")
    assert response.status_code == 200
    assert response.json() == [{"name": "test_model", "type": "test_type"}]

def test_load_model(mock_model_manager):
    response = client.post("/load_model?model_name=new_model&model_type=test_type")
    assert response.status_code == 200
    assert response.json() == {"message": "Model new_model loaded successfully"}
    mock_model_manager.load_model.assert_called_once_with("new_model", "test_type")

def test_load_model_error(mock_model_manager):
    mock_model_manager.load_model.side_effect = ValueError("Invalid model type")
    response = client.post("/load_model?model_name=new_model&model_type=invalid_type")
    assert response.status_code == 400
    assert "Invalid model type" in response.json()["detail"]
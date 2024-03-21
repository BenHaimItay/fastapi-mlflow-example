from fastapi.testclient import TestClient
from fastapi import FastAPI
import pytest
import numpy as np
from unittest.mock import AsyncMock, patch, MagicMock
from app.routes.v1.inference_router import router  # import your actual router

# Create a FastAPI instance and include the router under test
app = FastAPI()
app.include_router(router)


@pytest.fixture
def client():
    yield TestClient(app)


@pytest.mark.asyncio
@patch("app.routes.v1.inference_router.load_model", new_callable=AsyncMock)
async def test_predict_success(mock_load_model, client):
    # Given
    model_name = "some_model"
    model_version = "1"
    test_data = {"data": [1, 2, 3, 4]}

    # Mock objects
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0])
    mock_load_model.return_value = mock_model

    # When
    response = client.post(f"/{model_name}/{model_version}/predict", json=test_data)

    # Then
    assert response.status_code == 200
    assert response.json() == {"prediction": [0]}
    mock_model.predict.assert_called_once()

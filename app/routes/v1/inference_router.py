import logging
import mlflow.pyfunc
import uvicorn

from app.model_manager import load_model
from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from starlette import status

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/{model_name}/{model_version}/predict")
async def predict(model_name: str, model_version: str, payload: dict):
    try:
        model = await load_model(model_name, model_version)
        prediction = model.predict([payload["data"]])
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction error")

    return JSONResponse({"prediction": prediction.tolist()})

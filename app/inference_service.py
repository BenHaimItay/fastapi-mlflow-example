import logging
import mlflow.pyfunc
import uvicorn

from fastapi import FastAPI, HTTPException, APIRouter
from fastapi.responses import JSONResponse
from starlette import status
from typing import Any
from contextlib import asynccontextmanager

from app.config import settings
from app.routes.v1.inference_router import router as inference_router


logger = logging.getLogger(__name__)
general_router = APIRouter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    yield
    logger.info("Shutting down...")


@general_router.get("/health-check")
def health_check() -> Any:
    return JSONResponse(
        content={"message": "Inference service is strong and healthy"},
        status_code=status.HTTP_200_OK,
    )


class InferenceService:
    def __init__(self, title="InferenceService") -> None:
        self.app = FastAPI()
        self.model_cache = {}
        self._init_mlflow()
        self._init_routes()

    def _init_mlflow(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    def _init_routes(self) -> None:
        self.app.include_router(inference_router, tags=["inference_v1"])

    def listen(self) -> None:
        logger.info(f"App is running on port {settings.port}")
        uvicorn.run(
            self.app,
            port=int(settings.port),
            host=settings.host,
            loop="asyncio",
        )

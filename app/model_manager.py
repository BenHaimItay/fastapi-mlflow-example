import logging
import mlflow

from fastapi.exceptions import HTTPException

logger = logging.getLogger(__name__)
model_cache = {}


async def load_model(model_name, model_version):
    model_uri = f"models:/{model_name}/{model_version}"
    if model_uri in model_cache:
        logger.info(f"Model {model_uri} is already in cache")
        model = model_cache[model_uri]
    else:
        try:
            logger.info(f"Model {model_uri} is not in cache. Loading...")
            model = mlflow.pyfunc.load_model(model_uri)
            model_cache[model_uri] = model
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise HTTPException(status_code=500, detail="Model loading error")
    return model

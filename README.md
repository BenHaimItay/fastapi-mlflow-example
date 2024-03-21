# FastAPI MLflow integration


## Start MLFLOW server
```
poetry run mlflow server --host 127.0.0.1 --port 8001
```


## Start FastAPI server
```bash
poetry run python main.py
```
or
```bash
poetry run uvicorn main:app
```

## Train a model and log it to mlflow
```python
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

mlflow.set_tracking_uri("http://localhost:8001")
mlflow.set_experiment("iris-classification-model")

# Train your model
iris = load_iris()
params = {"max_depth": 4, "random_state": 42}
rf = RandomForestClassifier(**params)
rf.fit(iris.data, iris.target)

# Log the model
with mlflow.start_run() as run:
    mlflow.log_params(params)
    mlflow.log_metrics({"mse": mean_squared_error([17, 18, 19], [18, 18, 18])})
    mlflow.sklearn.log_model(
        sk_model=rf,
        registered_model_name="iris_rf_model",
        artifact_path="sklearn-model",
    )
```

## Send inference to the FastAPI-inference service
```python
import requests

# Replace these variables with your actual model's name and version.
model_name = "iris_rf_model"
model_version = 1  # Make sure this is the version you logged with MLflow
data = {"data": [5.1, 3.5, 1.4, 0.2]}  # An example input vector

# Make the prediction
response = requests.post(
    f"http://127.0.0.1:8000/{model_name}/{model_version}/predict",
    json=data,
)
print(response.json())
```

This will load the model from mlflow, cache it ( here is a future issue :) ) and make the prediction.

# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /code

# Copy the current directory contents into the container at /code
COPY ./app /code/app

# Install poetry and copy the lock file and pyproject.toml
RUN pip install poetry
COPY poetry.lock pyproject.toml /code/

RUN poetry install

# Define environment variable
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:8001

# Run app.py when the container launches
CMD ["poetry", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

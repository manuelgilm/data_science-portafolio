FROM python:3.12

# Set the working directory
RUN pip install --upgrade pip
RUN pip install mlflow
CMD ["sh", "-c", "mlflow server --backend-store-uri sqlite:///./mlflow/backend/mlflow.db --host 0.0.0.0 --port 5000"]

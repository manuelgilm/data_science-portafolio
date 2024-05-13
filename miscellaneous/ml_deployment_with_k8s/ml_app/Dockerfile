FROM ghcr.io/mlflow/mlflow
COPY . .
RUN python model.py
CMD ["mlflow", "models", "serve", "--model-uri","models:/DummyModel/latest","--port","5000","--host", "0.0.0.0", "--no-conda"]

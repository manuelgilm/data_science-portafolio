FROM python:3.11.0-slim-buster

# Set the working directory
RUN mkdir -p /train_service
WORKDIR /train_service
COPY ./train_service /train_service

# Install dependencies
RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry config virtualenvs.create true 
RUN poetry install

# Run the application
CMD ["poetry", "run", "fastapi", "dev", "train_service","--host", "0.0.0.0", "--port", "8000"]


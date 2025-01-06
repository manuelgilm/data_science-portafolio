FROM python:3.11
RUN mkdir /ml_service2
WORKDIR /ml_service2
COPY . /ml_service2
ENV PYTHONPATH=${PYTHONPATH}:${PWD}
RUN pip3 install poetry 
RUN poetry install

CMD ["poetry", "run", "fastapi", "dev","ml_service2", "--host", "0.0.0.0", "--port", "8002"]

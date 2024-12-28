FROM python:3.9 
RUN mkdir /app
WORKDIR /app
COPY . /app
ENV PYTHONPATH=${PYTHONPATH}:${PWD}
RUN pip3 install poetry
RUN poetry install 
CMD ["poetry", "run", "fastapi", "dev", "app", "--host", "0.0.0.0", "--port","5000"]
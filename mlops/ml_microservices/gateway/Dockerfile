FROM python:3.11
RUN mkdir /gateway
WORKDIR /gateway 
COPY . /gateway
ENV PYTHONPATH=${PYTHONPATH}:${PWD}
RUN pip3 install poetry 
RUN poetry install

CMD ["poetry", "run", "fastapi", "dev","gateway","--host", "0.0.0.0", "--port", "8000"]


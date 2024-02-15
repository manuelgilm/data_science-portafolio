FROM python:3.9
RUN mkdir /iris_classifier
WORKDIR /iris_classifier
COPY . /iris_classifier
ENV PYTHONPATH=${PYTHONPATH}:${PWD}
RUN pip3 install poetry 
RUN poetry install  

CMD ["poetry", "run", "python", "run.py"]

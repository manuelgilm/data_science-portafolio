FROM python:3.9
RUN mkdir /credit_approval
WORKDIR /credit_approval
COPY . /credit_approval
ENV PYTHONPATH=${PYTHONPATH}:${PWD}
RUN pip3 install poetry 
RUN poetry install  

CMD ["poetry", "run", "python", "run.py"]

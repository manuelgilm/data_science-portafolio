FROM python:3.9
RUN mkdir /ml_monitor
WORKDIR /ml_monitor
COPY . /ml_monitor
ENV PYTHONPATH=${PYTHONPATH}:${PWD}
RUN pip3 install poetry 
RUN poetry install  
EXPOSE 5000
CMD ["poetry", "run", "python", "ml_monitor/app.py", "--host=0.0.0.0", "--port=5000"]

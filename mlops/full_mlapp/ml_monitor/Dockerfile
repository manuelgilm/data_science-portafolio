FROM python:3.9
RUN mkdir /ml_monitor
WORKDIR /ml_monitor
COPY . /ml_monitor
ENV PYTHONPATH=${PYTHONPATH}:${PWD}
RUN pip3 install poetry 
RUN poetry install  
EXPOSE 8501
CMD ["poetry", "run", "python","-m", "streamlit","run", "ml_monitor/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501","--server.enableCORS=false", "--server.enableXsrfProtection=false"]

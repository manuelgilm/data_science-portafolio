FROM python:3.9.1

WORKDIR /opt/demo/
COPY /app .

RUN pip install -r requirements.txt

ENTRYPOINT python app.py
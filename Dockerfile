FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y wget mesa-utils libglib2.0-0 libsm6 libxrender1 libxext6
RUN wget https://github.com/jwilder/dockerize/releases/download/v0.6.1/dockerize-linux-amd64-v0.6.1.tar.gz
RUN tar -xvzf dockerize-linux-amd64-v0.6.1.tar.gz -C /usr/local/bin

ENV PYTHONPATH="/app"

COPY utils /app/utils
COPY ml /app/ml
COPY server /app/server

CMD /usr/local/bin/dockerize -wait tcp://postgres:5432 -timeout 30s && cd server && python create_db.py && uvicorn main:app --host 0.0.0.0 --port 8000

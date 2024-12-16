FROM tensorflow/tensorflow:2.10.0
FROM python:3.10-slim

WORKDIR /prod

COPY brainmap /prod/brainmap

COPY .env /prod/.env

COPY requirements.txt /prod/requirements.txt

RUN pip install --upgrade pip && pip install --no-cache-dir -r /prod/requirements.txt

ENV PORT=8000

CMD uvicorn brainmap.api.fast:app --host 0.0.0.0 --port $PORT

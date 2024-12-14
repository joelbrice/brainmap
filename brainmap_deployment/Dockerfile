FROM python:3.10-slim

WORKDIR /app

COPY brainmap_deployment /app/brainmap_deployment
COPY .env /app/.env
COPY requirements.txt /app/requirements.txt


RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8000"]

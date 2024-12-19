FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6

WORKDIR /prod

COPY brainmap /prod/brainmap

COPY .env /prod/.env

COPY requirements.txt /prod/requirements.txt

RUN pip install --upgrade pip && pip install --no-cache-dir -r /prod/requirements.txt

ENV PORT=8000

CMD ["uvicorn", "brainmap.api.fast:app", "--host", "0.0.0.0", "--port", "$PORT"]

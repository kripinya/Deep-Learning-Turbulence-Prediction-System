# Dockerfile - production-like image for the Flask + Gunicorn app
FROM python:3.11-slim

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app code and model artifacts
COPY . /app

ENV PORT=8080
EXPOSE 8080

# Run Gunicorn with 2 workers
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "api.app:app", "--timeout", "120"]

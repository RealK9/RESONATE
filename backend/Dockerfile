FROM python:3.11-slim

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 libfftw3-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps — base first (cached layer), then ML
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ../requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# App code
COPY . .

# Persistent volumes for samples + ML data
VOLUME ["/app/samples", "/app/vector_indexes", "/data"]

# Pre-create dirs
RUN mkdir -p uploads transposed_cache vector_indexes

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "server.py"]

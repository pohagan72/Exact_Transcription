# Use an official Python runtime as a parent image (choose a specific version)
FROM python:3.10-slim

# Set environment variables to prevent interactive prompts during build
ENV PYTHONUNBUFFERED True
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies including ffmpeg and git (sometimes needed for pip installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
# Ensure requirements.txt includes: flask, gunicorn, python-dotenv, requests,
# whisperx, torch, torchaudio, soundfile, pyannote.audio==VERSION_YOU_NEED, numpy, werkzeug,
# huggingface_hub==VERSION_YOU_NEED, psutil, pydub, tqdm, ctranslate2
# Pin versions where possible for reproducibility!
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port 8080 available to the world outside this container
# Cloud Run expects containers to listen on the port defined by the PORT env var, defaulting to 8080.
EXPOSE 8080

# Use a production-grade WSGI server like Gunicorn
# It reads the PORT environment variable provided by Cloud Run.
# Ensure 'main:app' matches your Python filename (main.py) and Flask app object name (app).
# Increase workers/threads based on testing and instance size. Increase timeout for long processing.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 1200 main:app

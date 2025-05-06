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
# huggingface_hub==VERSION_YOU_NEED, psutil, pydub, tqdm, ctranslate2, gevent
# Pin versions where possible for reproducibility!
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make port available to the world outside this container
# Cloud Run expects containers to listen on the port defined by the PORT env var, defaulting to 8080.
# The application (main.py) will now read the PORT environment variable.
EXPOSE 8080

# Run the Python application directly using gevent's WSGIServer (if FLASK_DEBUG is not true)
# or Flask's dev server (if FLASK_DEBUG is true) as defined in main.py's __main__ block.
CMD ["python", "main.py"]
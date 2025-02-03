import os
import time
import logging
import psutil
import torch
import whisperx
import soundfile as sf
import librosa
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, g, current_app
from werkzeug.utils import secure_filename
from pyannote.audio import Pipeline
import threading
from flask import copy_current_request_context
from pydub import AudioSegment  # New import for conversion

# --------------------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# CUDA Device Detection
# --------------------------------------------------------------------------
def get_cuda_device():
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA is available. Using GPU.")
    else:
        device = "cpu"
        logger.info("CUDA is not available. Using CPU.")
    return device

# --------------------------------------------------------------------------
# Convert Input Audio to WAV (if necessary)
# --------------------------------------------------------------------------
def convert_to_wav(input_path):
    try:
        file_root, file_ext = os.path.splitext(input_path)
        wav_path = file_root + ".wav"
        audio = AudioSegment.from_file(input_path)  # This uses FFmpeg in the background.
        audio.export(wav_path, format="wav")
        logger.info(f"Converted {input_path} to WAV format: {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"Error converting file to WAV: {str(e)}")
        raise

# --------------------------------------------------------------------------
# Audio Chunking Function
# --------------------------------------------------------------------------
def chunk_audio(audio_path, chunk_duration=300):  # Chunk duration in seconds (5 minutes)
    try:
        y, sr = librosa.load(audio_path, sr=None)
        chunk_size = chunk_duration * sr
        chunks = []
        for i in range(0, len(y), chunk_size):
            chunk = y[i:i + chunk_size]
            chunk_filename = os.path.join(UPLOAD_FOLDER, f"chunk_{i // chunk_size}.wav")
            sf.write(chunk_filename, chunk, sr)  # Write the chunk as proper WAV
            chunks.append(chunk_filename)
        return chunks
    except Exception as e:
        logger.error(f"Error during audio chunking: {str(e)}")
        raise

# --------------------------------------------------------------------------
# System Usage Logging Function
# --------------------------------------------------------------------------
def log_system_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_usage = psutil.virtual_memory().percent
    logger.info(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_usage}%")

# --------------------------------------------------------------------------
# Allowed File Type Check
# --------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --------------------------------------------------------------------------
# Flask App Configuration
# --------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "replace_this_with_a_secure_random_string")
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB

UPLOAD_FOLDER = 'temp'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------------------------------------------------------
# Model Loading and Reuse in Request Context
# --------------------------------------------------------------------------
def load_whisper_model(device):
    model = whisperx.load_model("large-v2", device)
    logger.info("Whisper model loaded.")
    return model

def load_diarization_pipeline(device):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token="HF_Token_Goes_Here"
    ).to(torch.device(device))
    logger.info("Diarization pipeline loaded.")
    return pipeline

# --------------------------------------------------------------------------
# Custom Timeout Mechanism
# --------------------------------------------------------------------------
class TimeoutError(Exception):
    pass

def timeout(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target)
            thread.start()
            thread.join(seconds)

            if thread.is_alive():
                raise TimeoutError(f"Function timed out after {seconds} seconds")
            if exception[0] is not None:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator

# --------------------------------------------------------------------------
# Main Function for Transcription and Diarization
# --------------------------------------------------------------------------
@timeout(3600)  # 1-hour timeout
def transcribe_and_diarize(audio_file_path, device, whisper_model, diarization_pipeline):
    log_system_usage()

    chunks = chunk_audio(audio_file_path)
    full_transcript = []

    try:
        for chunk_path in chunks:
            logger.info(f"Processing chunk: {chunk_path}")
            audio = whisperx.load_audio(chunk_path)
            result = whisper_model.transcribe(audio, batch_size=16)

            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device)

            diarization = diarization_pipeline(chunk_path)

            for segment in result["segments"]:
                segment_start = segment["start"]
                segment_end = segment["end"]
                speaker_times = {}
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if turn.start <= segment_end and turn.end >= segment_start:
                        overlap = min(turn.end, segment_end) - max(turn.start, segment_start)
                        if overlap > 0:
                            speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap
                dominant_speaker = max(speaker_times.items(), key=lambda x: x[1])[0] if speaker_times else "Unknown Speaker"
                formatted_segment = f"[{segment_start:.2f}-{segment_end:.2f}] {dominant_speaker}: {segment['text']}"
                full_transcript.append(formatted_segment)
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise RuntimeError(f"Error during processing: {str(e)}")
    finally:
        # Clean up chunk files
        for chunk_path in chunks:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    return "\n".join(full_transcript)

# --------------------------------------------------------------------------
# Flask Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    transcript = None
    whisper_model = None
    diarization_pipeline = None
    if request.method == "POST":
        if "audio_file" not in request.files:
            flash("No file selected!")
            return redirect(request.url)
        file = request.files["audio_file"]
        if file.filename == "":
            flash("No file selected!")
            return redirect(request.url)
        if not allowed_file(file.filename):
            flash("Invalid file type! Allowed types are: mp3, mp4, mpeg, mpga, m4a, wav, webm")
            return redirect(request.url)

        filename = secure_filename(file.filename)
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        try:
            # Load models in the context of the request
            device = get_cuda_device()
            whisper_model = load_whisper_model(device)
            diarization_pipeline = load_diarization_pipeline(device)

            # If the file is not a WAV, convert it first.
            processed_path = temp_path
            if not filename.lower().endswith(".wav"):
                processed_path = convert_to_wav(temp_path)

            transcript = transcribe_and_diarize(processed_path, device, whisper_model, diarization_pipeline)
        except TimeoutError:
            transcript = "Processing timed out. Please try again with a smaller file."
        except Exception as e:
            transcript = f"An error occurred: {str(e)}"
        finally:
            # Remove both the original and converted files if they exist.
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if processed_path != temp_path and os.path.exists(processed_path):
                os.remove(processed_path)

    return render_template("index.html", transcript=transcript)

# --------------------------------------------------------------------------
# Run the Flask App
# --------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

import os
import time
import logging
import psutil
import torch
import whisperx
import soundfile as sf
import librosa
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
from werkzeug.utils import secure_filename
from pyannote.audio import Pipeline
from pydub import AudioSegment  # For audio conversion
from tqdm import tqdm  # Import tqdm for progress tracking
import io  # Import the io module
import ctranslate2
import uuid

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
        audio = AudioSegment.from_file(input_path)
        audio.export(wav_path, format="wav")
        logger.info(f"Converted {input_path} to WAV format: {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"Error converting file to WAV: {str(e)}")
        raise

# --------------------------------------------------------------------------
# Audio Chunking Function
# --------------------------------------------------------------------------
def chunk_audio(audio_path, chunk_duration=300):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        chunk_size = chunk_duration * sr
        chunks = []
        for i in range(0, len(y), chunk_size):
            chunk = y[i:i + chunk_size]
            chunk_filename = os.path.join(UPLOAD_FOLDER, f"chunk_{i // chunk_size}.wav")
            sf.write(chunk_filename, chunk, sr)
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
    try:
        # Attempt to load the model with float16 first
        model = whisperx.load_model("large-v2", device, compute_type="float16")
    except ValueError:
        # If float16 is not supported, fall back to float32
        logger.warning("Float16 computation is not supported. Using float32 instead.")
        model = whisperx.load_model("large-v2", device, compute_type="float32")
    logger.info("Whisper model loaded.")
    return model

def load_diarization_pipeline(device):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token="Key_Goes_Here"
    ).to(torch.device(device))
    logger.info("Diarization pipeline loaded.")
    return pipeline

# --------------------------------------------------------------------------
# Main Function for Transcription and Diarization with Progress Tracking
# --------------------------------------------------------------------------
def transcribe_and_diarize(audio_file_path, device, whisper_model, diarization_pipeline):
    log_system_usage()

    chunks = chunk_audio(audio_file_path)
    full_transcript = []

    try:
        for chunk_path in tqdm(chunks, desc="Processing Chunks", unit="chunk"):
            logger.info(f"Processing chunk: {chunk_path}")
            audio = whisperx.load_audio(chunk_path)
            result = whisper_model.transcribe(audio, batch_size=16)
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            del model_a
            torch.cuda.empty_cache()
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
        for chunk_path in chunks:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)

    return "\n".join(full_transcript)

# --------------------------------------------------------------------------
# Flask Routes
# --------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audio_file" not in request.files:
            flash("No file selected!", "error")  # Use a category for flash messages
            return redirect(request.url)

        audio_file = request.files["audio_file"]
        if audio_file.filename == "":
            flash("No file selected!", "error")
            return redirect(request.url)

        if not allowed_file(audio_file.filename):
            flash("Invalid file type! Allowed types are: mp3, mp4, mpeg, mpga, m4a, wav, webm", "error")
            return redirect(request.url)

        try:
            filename = secure_filename(audio_file.filename)
            temp_audio_path = os.path.join(UPLOAD_FOLDER, filename)
            audio_file.save(temp_audio_path)

            # Get the duration of the audio file
            duration = librosa.get_duration(filename=temp_audio_path)
            duration_minutes = int(duration // 60)
            duration_seconds = int(duration % 60)

            device = get_cuda_device()
            whisper_model = load_whisper_model(device)
            diarization_pipeline = load_diarization_pipeline(device)

            processed_audio_path = temp_audio_path
            if not filename.lower().endswith(".wav"):
                processed_audio_path = convert_to_wav(temp_audio_path)

            transcript = transcribe_and_diarize(processed_audio_path, device, whisper_model, diarization_pipeline)
            transcript_id = str(uuid.uuid4())
            transcript_path = os.path.join(UPLOAD_FOLDER, f"{transcript_id}.txt")
            with open(transcript_path, 'w') as f:
                f.write(transcript)
            session['transcript_id'] = transcript_id

            flash(f"File '{filename}' selected. Duration: {duration_minutes} minutes {duration_seconds} seconds.", "info")
            return redirect(url_for('download_transcript'))

        except RuntimeError as e:
            flash(f"An error occurred during processing: {str(e)}", "error")
            logger.exception("Transcription/Diarization Error:") # Log the full exception
        except Exception as e:
            flash(f"An unexpected error occurred: {str(e)}", "error")
            logger.exception("Unexpected Error:") # Log the full exception
        finally:
            # Clean up temporary files, regardless of success or failure
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
            if 'processed_audio_path' in locals() and processed_audio_path != temp_audio_path and os.path.exists(processed_audio_path):
                os.remove(processed_audio_path)

    return render_template("index.html")

@app.route("/download")
def download_transcript():
    transcript_id = session.get('transcript_id', None)
    if transcript_id:
        transcript_path = os.path.join(UPLOAD_FOLDER, f"{transcript_id}.txt")
        if os.path.exists(transcript_path):
            return send_file(
                transcript_path,
                mimetype="text/plain",
                as_attachment=True,
                download_name="transcript.txt"
            )
        else:
            flash("Transcript file not found.", "error")
    else:
        flash("No transcript available for download.", "error")
    return redirect(url_for('index'))

# --------------------------------------------------------------------------
# Run the Flask App
# --------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)

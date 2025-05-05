# --- Standard Library Imports ---
import os
import time
import logging
import uuid
import io      # Keep for now, might be needed by dependencies
import json    # Needed for SSE route data formatting

# --- Third-party Library Imports ---
# Flask & Web Related
from flask import (
    Flask, request, jsonify, render_template, url_for, send_file, session, Response
)
from werkzeug.utils import secure_filename # Needed for file uploads

# Celery Task Queue
from celery import Celery, current_task
from celery.utils.log import get_task_logger # <<< IMPORT THIS

# ML / Audio Processing
import psutil
import torch
import whisperx
import soundfile as sf
import librosa
import numpy as np
from pyannote.audio import Pipeline
from pydub import AudioSegment
from tqdm import tqdm
import ctranslate2

# Utilities
from dotenv import load_dotenv

# --------------------------------------------------------------------------
# Load Environment Variables & Logging Configuration
# --------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Define the GLOBAL logger (used outside Celery tasks)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Constants and Configuration
# --------------------------------------------------------------------------
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "default_insecure_secret_key_please_change")
if FLASK_SECRET_KEY == "default_insecure_secret_key_please_change":
    logger.warning("FLASK_SECRET_KEY not set in environment. Using default (INSECURE). Set in .env file.")

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024
AUDIO_CHUNK_DURATION_S = 300

# --------------------------------------------------------------------------
# Flask App Initialization
# --------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_BYTES
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --------------------------------------------------------------------------
# Celery Configuration
# --------------------------------------------------------------------------
app.config['broker_url'] = 'redis://localhost:6379/0'
app.config['result_backend'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['broker_url'])
celery.conf.update(app.config)


# --------------------------------------------------------------------------
# ORIGINAL HELPER FUNCTIONS (Use GLOBAL logger)
# --------------------------------------------------------------------------
def get_cuda_device():
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA is available. Using GPU.")
    else:
        device = "cpu"
        logger.info("CUDA is not available. Using CPU.")
    return device

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_system_usage(context="General"):
    try:
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        logger.info(f"[{context}] CPU Usage: {cpu_usage:.1f}%, Memory Usage: {memory_usage:.1f}% ({memory_info.used/1024**3:.2f}/{memory_info.total/1024**3:.2f} GB)")
    except Exception as e:
        logger.warning(f"Could not log system usage: {e}")

def convert_to_wav(input_path, output_folder):
    try:
        file_root, file_ext = os.path.splitext(os.path.basename(input_path))
        wav_filename = f"{file_root}_converted.wav"
        wav_path = os.path.join(output_folder, wav_filename)
        logger.info(f"Attempting to convert {input_path} to WAV...")
        audio = AudioSegment.from_file(input_path)
        audio.export(wav_path, format="wav")
        logger.info(f"Successfully converted to WAV: {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"Error converting file '{input_path}' to WAV: {e}", exc_info=True)
        raise RuntimeError(f"Failed to convert file to WAV: {e}") from e

def chunk_audio(audio_path, output_folder, chunk_duration_s=AUDIO_CHUNK_DURATION_S):
    chunks = []
    num_chunks = 0
    try:
        logger.info(f"Loading audio for chunking: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None, mono=False)
        if y.ndim > 1 and y.shape[0] > 1:
            y = librosa.to_mono(y)
            logger.info("Converted audio to mono for chunking.")

        total_duration_s = librosa.get_duration(y=y, sr=sr)
        chunk_size_samples = int(chunk_duration_s * sr)
        num_chunks = int(np.ceil(total_duration_s / chunk_duration_s)) if total_duration_s > 0 else 0

        logger.info(f"Chunking audio into ~{num_chunks} chunks of {chunk_duration_s}s each.")
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]

        for i in range(0, len(y), chunk_size_samples):
            chunk_num = i // chunk_size_samples
            chunk_start_sample = i
            chunk_end_sample = min(i + chunk_size_samples, len(y))
            chunk_audio_data = y[chunk_start_sample:chunk_end_sample]

            chunk_filename = os.path.join(output_folder, f"{base_filename}_chunk_{chunk_num:03d}.wav")
            logger.debug(f"Writing chunk {chunk_num+1}/{num_chunks} to {chunk_filename}")
            sf.write(chunk_filename, chunk_audio_data, sr)
            chunks.append(chunk_filename)

        logger.info(f"Successfully created {len(chunks)} audio chunks.")
        return chunks, num_chunks
    except Exception as e:
        logger.error(f"Error during audio chunking for '{audio_path}': {e}", exc_info=True)
        for chunk_file in chunks:
            if os.path.exists(chunk_file):
                try: os.remove(chunk_file)
                except OSError: logger.warning(f"Could not clean up chunk file: {chunk_file}")
        raise RuntimeError(f"Failed to chunk audio: {e}") from e

def safe_cleanup(*filepaths):
    for filepath in filepaths:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up temporary file: {filepath}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {filepath}: {e}")


# --------------------------------------------------------------------------
# ORIGINAL Model Loading Functions (Use GLOBAL logger)
# --------------------------------------------------------------------------
def load_whisper_model(device):
    compute_type = "float16" if device == "cuda" else "int8"
    model = None
    try:
        logger.info(f"Loading Whisper large-v2 model with compute_type={compute_type} on device={device}...")
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        logger.info("Whisper model loaded successfully.")
    except (ValueError, RuntimeError) as e:
        logger.warning(f"{compute_type} computation failed or not supported ({e}). Trying float32...")
        compute_type = "float32"
        try:
            model = whisperx.load_model("large-v2", device, compute_type=compute_type)
            logger.info("Whisper model loaded successfully with compute_type=float32.")
        except Exception as e_fallback:
            logger.error(f"Failed to load Whisper model even with float32: {e_fallback}", exc_info=True)
            raise RuntimeError(f"Could not load Whisper model: {e_fallback}") from e_fallback
    except Exception as e_general:
        logger.error(f"An unexpected error occurred loading Whisper model: {e_general}", exc_info=True)
        raise RuntimeError(f"Could not load Whisper model: {e_general}") from e_general
    return model

def load_diarization_pipeline(device):
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        logger.error("HUGGING_FACE_HUB_TOKEN environment variable not set. Cannot load diarization model.")
        raise ValueError("Missing HUGGING_FACE_HUB_TOKEN. Please set it in the .env file or environment.")
    pipeline = None
    try:
        logger.info("Loading Pyannote speaker-diarization-3.0 pipeline...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=hf_token)
        pipeline.to(torch.device(device))
        logger.info("Diarization pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load diarization pipeline: {e}", exc_info=True)
        logger.error("Ensure you have accepted the user agreement for pyannote/speaker-diarization-3.0 and segmentation-3.0 on Hugging Face Hub.")
        raise RuntimeError(f"Could not load diarization pipeline: {e}") from e
    return pipeline


# --------------------------------------------------------------------------
# MODIFIED Core Transcription and Diarization Logic (Use GLOBAL logger)
# --------------------------------------------------------------------------
def transcribe_and_diarize(audio_path, device, whisper_model, diarization_pipeline, progress_callback=None):
    log_system_usage(f"Start Processing: {os.path.basename(audio_path)}")
    if progress_callback: progress_callback({'status': 'Analyzing audio...', 'percent': 18})

    full_transcript_lines = []
    chunk_files_to_process = []
    chunk_files_to_cleanup = []
    original_duration = 0
    num_chunks = 1

    try:
        original_duration = librosa.get_duration(path=audio_path)
        logger.info(f"Original audio duration: {original_duration:.2f} seconds.")

        if original_duration > AUDIO_CHUNK_DURATION_S * 1.1:
            logger.info("Audio is long, chunking...")
            if progress_callback: progress_callback({'status': 'Chunking audio...', 'percent': 19})
            # chunk_audio uses global logger
            created_chunks, num_chunks = chunk_audio(audio_path, app.config['UPLOAD_FOLDER'])
            if not created_chunks: raise RuntimeError("Audio chunking resulted in no files.")
            chunk_files_to_process = created_chunks
            chunk_files_to_cleanup = created_chunks
        else:
            logger.info("Audio is short enough, processing as a single file.")
            chunk_files_to_process = [audio_path]

        if progress_callback: progress_callback({'status': 'Starting transcription...', 'percent': 20})

        total_offset = 0.0
        cumulative_processed_duration = 0.0

        for i, chunk_path in enumerate(chunk_files_to_process):
            current_chunk_num = i + 1
            log_system_usage(f"Start Chunk {current_chunk_num}/{num_chunks}")
            logger.info(f"Processing chunk {current_chunk_num}/{num_chunks}: {os.path.basename(chunk_path)}")

            base_percent = 20
            chunk_progress_span = 70
            percent_start_chunk = base_percent + int(chunk_progress_span * (i / num_chunks)) if num_chunks > 0 else base_percent

            try:
                chunk_audio_data = whisperx.load_audio(chunk_path)
                try: chunk_duration = librosa.get_duration(path=chunk_path)
                except Exception:
                    chunk_duration = len(chunk_audio_data) / 16000.0 if len(chunk_audio_data) > 0 else AUDIO_CHUNK_DURATION_S
                    logger.warning(f"Using estimated duration for chunk {chunk_path}")

                # 1. Transcription
                if progress_callback: progress_callback({'status': f'Transcribing chunk {current_chunk_num}/{num_chunks}...', 'percent': percent_start_chunk + 5})
                logger.debug("Running Whisper transcription...")
                result = whisper_model.transcribe(chunk_audio_data, batch_size=16)
                if not result or "segments" not in result or not result["segments"]:
                    logger.warning(f"Whisper produced no segments for chunk: {os.path.basename(chunk_path)}")
                    total_offset += chunk_duration
                    cumulative_processed_duration += chunk_duration
                    if progress_callback:
                        percent_end_chunk = base_percent + int(chunk_progress_span * ((i + 1) / num_chunks)) if num_chunks > 0 else (base_percent + chunk_progress_span)
                        progress_callback({'status': f'Chunk {current_chunk_num}/{num_chunks} processed (no speech).', 'percent': percent_end_chunk})
                    continue
                logger.debug(f"Transcription found {len(result['segments'])} segments.")

                # 2. Alignment
                if progress_callback: progress_callback({'status': f'Aligning chunk {current_chunk_num}/{num_chunks}...', 'percent': percent_start_chunk + 10})
                aligned_result = result
                try:
                    logger.debug("Loading alignment model...")
                    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
                    logger.debug("Running alignment...")
                    aligned_result = whisperx.align(result["segments"], align_model, metadata, chunk_audio_data, device, return_char_alignments=False)
                    del align_model
                    if device == 'cuda': torch.cuda.empty_cache()
                    logger.debug("Alignment complete.")
                except Exception as align_error:
                    logger.warning(f"Alignment failed for chunk {os.path.basename(chunk_path)}: {align_error}. Using original Whisper timestamps.")

                # 3. Diarization
                if progress_callback: progress_callback({'status': f'Identifying speakers chunk {current_chunk_num}/{num_chunks}...', 'percent': percent_start_chunk + 15})
                diarization = None
                try:
                    logger.debug("Running speaker diarization...")
                    diarization = diarization_pipeline(chunk_path)
                    logger.debug("Diarization complete.")
                except Exception as diarize_error:
                    logger.error(f"Diarization failed for chunk {os.path.basename(chunk_path)}: {diarize_error}", exc_info=True)

                # 4. Assign Speakers
                if progress_callback: progress_callback({'status': f'Assigning speakers chunk {current_chunk_num}/{num_chunks}...', 'percent': percent_start_chunk + 20})
                for segment in aligned_result["segments"]:
                    seg_start, seg_end, seg_text = segment.get("start"), segment.get("end"), segment.get("text", "").strip()
                    if seg_start is None or seg_end is None or not seg_text:
                        logger.debug(f"Skipping segment with missing start/end/text: {segment}")
                        continue
                    global_start, global_end = total_offset + seg_start, total_offset + seg_end
                    dominant_speaker = "Unknown Speaker"
                    if diarization:
                        speaker_times = {}
                        try:
                            for turn, _, speaker in diarization.itertracks(yield_label=True):
                                overlap_start = max(turn.start, seg_start)
                                overlap_end = min(turn.end, seg_end)
                                overlap_duration = overlap_end - overlap_start
                                if overlap_duration > 0.01: speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap_duration
                            if speaker_times: dominant_speaker = max(speaker_times, key=speaker_times.get)
                        except Exception as speaker_assign_err:
                             logger.warning(f"Error assigning speaker to segment [{seg_start:.2f}-{seg_end:.2f}]: {speaker_assign_err}")
                    formatted_line = f"[{global_start:08.2f}-{global_end:08.2f}] {dominant_speaker}: {seg_text}"
                    full_transcript_lines.append(formatted_line)

                total_offset += chunk_duration
                cumulative_processed_duration += chunk_duration
                log_system_usage(f"End Chunk {current_chunk_num}/{num_chunks}")
                logger.debug(f"Cumulative processed duration: {cumulative_processed_duration:.2f}s")

                if progress_callback:
                    percent_end_chunk = base_percent + int(chunk_progress_span * ((i + 1) / num_chunks)) if num_chunks > 0 else (base_percent + chunk_progress_span)
                    progress_callback({'status': f'Chunk {current_chunk_num}/{num_chunks} complete.', 'percent': percent_end_chunk})

            except Exception as chunk_proc_error:
                 logger.error(f"Failed to process chunk {os.path.basename(chunk_path)}: {chunk_proc_error}", exc_info=True)
                 if progress_callback:
                     percent_end_chunk = base_percent + int(chunk_progress_span * ((i + 1) / num_chunks)) if num_chunks > 0 else (base_percent + chunk_progress_span)
                     progress_callback({'status': f'Error on chunk {current_chunk_num}/{num_chunks}. Continuing...', 'percent': percent_end_chunk })
                 try:
                     chunk_dur = librosa.get_duration(path=chunk_path)
                     total_offset += chunk_dur
                     cumulative_processed_duration += chunk_dur
                 except Exception:
                     logger.warning(f"Could not get duration for failed chunk {chunk_path}. Offset might be inaccurate.")
                     total_offset += AUDIO_CHUNK_DURATION_S

    except Exception as e:
        logger.error(f"Fatal error during transcription/diarization process for {audio_path}: {e}", exc_info=True)
        raise RuntimeError(f"Processing failed: {e}") from e
    finally:
        if chunk_files_to_cleanup:
            logger.info("Cleaning up temporary audio chunks...")
            safe_cleanup(*chunk_files_to_cleanup)
        log_system_usage(f"End Processing: {os.path.basename(audio_path)}")

    if not full_transcript_lines:
         logger.warning(f"Processing completed but no transcript was generated for {audio_path}.")
         if progress_callback: progress_callback({'status': 'Processing complete (no speech detected).', 'percent': 95})
         return "[No speech detected or processing error resulted in empty transcript]"

    if progress_callback: progress_callback({'status': 'Finalizing transcript...', 'percent': 95})
    return "\n".join(full_transcript_lines)


# --------------------------------------------------------------------------
# CELERY TASK DEFINITION (Using CORRECT task logger)
# --------------------------------------------------------------------------
@celery.task(bind=True)
def process_audio_task(self, temp_audio_path, original_filename):
    """Celery task for background processing."""
    # Get the task-specific logger using celery.utils.log
    task_logger = get_task_logger(__name__) # <<< USE THIS FOR TASK LOGGING

    transcript_filename = f"{uuid.uuid4().hex}_transcript.txt"
    # Access Flask config via the app instance
    upload_dir = app.config['UPLOAD_FOLDER']
    transcript_path = os.path.join(upload_dir, transcript_filename)
    processed_audio_path = None
    task_id = self.request.id
    task_logger.info(f"Task {task_id}: Starting processing for {original_filename}") # Use task_logger

    # --- Define progress callback linked to this task instance ---
    def progress_callback(update_data):
        task_logger.info(f"Task {task_id} Progress: {update_data}") # Use task_logger
        try:
            update_data['task_id'] = task_id
            self.update_state(state='PROGRESS', meta=update_data)
        except Exception as e:
            # Log error in updating state using the task logger
            task_logger.error(f"Task {task_id}: Failed to update progress state: {e}")

    try:
        # 0. Report Initial State & Log System Usage
        progress_callback({'status': 'Preparing...', 'percent': 5})
        log_system_usage(f"Task {task_id} Prepare") # Uses global logger

        # 1. Conversion
        processed_audio_path = temp_audio_path
        if not original_filename.lower().endswith(".wav"):
             progress_callback({'status': 'Converting audio to WAV...', 'percent': 10})
             processed_audio_path = convert_to_wav(temp_audio_path, upload_dir) # Uses global logger
             progress_callback({'status': 'Conversion complete.', 'percent': 15})

        # 2. Load models
        progress_callback({'status': 'Loading models (if needed)...', 'percent': 16})
        device = get_cuda_device() # Uses global logger
        whisper_model_instance = load_whisper_model(device) # Uses global logger
        diarization_pipeline_instance = load_diarization_pipeline(device) # Uses global logger
        progress_callback({'status': 'Models ready.', 'percent': 17})

        # 4. Run Core Logic
        # transcribe_and_diarize uses global logger, takes progress_callback
        transcript_content = transcribe_and_diarize(
            processed_audio_path,
            device,
            whisper_model_instance,
            diarization_pipeline_instance,
            progress_callback=progress_callback
        )

        # 5. Save Transcript
        progress_callback({'status': 'Saving transcript...', 'percent': 98})
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_content)
        task_logger.info(f"Task {task_id}: Transcript saved to: {transcript_path}") # Use task_logger

        # 6. Report Completion
        final_meta = {
            'status': 'Complete',
            'percent': 100,
            'result_filename': transcript_filename
        }
        log_system_usage(f"Task {task_id} Complete") # Uses global logger
        return final_meta

    except Exception as e:
        error_message = f"Error processing {original_filename}: {str(e)}"
        task_logger.error(f"Task {task_id}: {error_message}", exc_info=True) # Use task_logger
        log_system_usage(f"Task {task_id} Failed") # Uses global logger
        self.update_state(
            state='FAILURE',
            meta={'status': error_message, 'percent': -1, 'exc_type': type(e).__name__, 'exc_message': str(e)}
        )
        # No re-raise needed, state is updated

    finally:
        # Cleanup uses global helper safe_cleanup (which uses global logger)
        task_logger.info(f"Task {task_id}: Running cleanup.") # Use task_logger for task context msg
        safe_cleanup(temp_audio_path)
        if processed_audio_path and processed_audio_path != temp_audio_path:
            safe_cleanup(processed_audio_path)

# --------------------------------------------------------------------------
# Flask Routes (Use GLOBAL logger)
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index_route():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio_file" not in request.files:
        logger.warning("Upload attempt with no file part.")
        return jsonify({"error": "No file part selected."}), 400
    audio_file = request.files["audio_file"]
    if audio_file.filename == "":
        logger.warning("Upload attempt with no filename.")
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(audio_file.filename):
        logger.warning(f"Upload attempt with invalid file type: {audio_file.filename}")
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(audio_file.filename)
    unique_id = uuid.uuid4().hex
    temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")

    try:
        logger.info(f"Receiving file: {filename}")
        audio_file.save(temp_audio_path)
        logger.info(f"File saved temporarily to: {temp_audio_path}")
        task = process_audio_task.delay(temp_audio_path, filename)
        logger.info(f"Delegated processing for {filename} to task ID: {task.id}")
        return jsonify({"task_id": task.id}), 202
    except Exception as e:
        logger.error(f"Error during upload processing for {filename}: {e}", exc_info=True)
        safe_cleanup(temp_audio_path)
        return jsonify({"error": f"Server error during upload: {str(e)}"}), 500

@app.route("/status/<task_id>")
def task_status(task_id):
    task = process_audio_task.AsyncResult(task_id)
    response_data = {'state': task.state, 'info': {'status': 'Unknown state', 'percent': 0}}
    if isinstance(task.info, dict): response_data['info'] = task.info
    elif task.info is not None: response_data['info']['status'] = f"Error: {str(task.info)}"
    # Refine status based on state
    if task.state == 'PENDING':
        response_data['info'].setdefault('status', 'Waiting in queue...')
        response_data['info'].setdefault('percent', 0)
    elif task.state == 'STARTED':
        response_data['info'].setdefault('status', 'Task started...')
        response_data['info'].setdefault('percent', 2)
    elif task.state == 'RETRY':
        response_data['info'].setdefault('status', f"Task retrying: {str(task.info)}")
        response_data['info'].setdefault('percent', 0)
    elif task.state == 'FAILURE':
        if not response_data['info'].get('status', '').startswith('Error') and not response_data['info'].get('status', '').startswith('Task failed'):
             response_data['info']['status'] = f"Task failed: {str(task.info)}"
        response_data['info']['percent'] = -1
    elif task.state == 'SUCCESS':
        response_data['info'].setdefault('status', 'Complete')
        response_data['info'].setdefault('percent', 100)
    return jsonify(response_data)

@app.route('/stream-progress/<task_id>')
def stream_progress(task_id):
    def generate():
        last_data_sent_str = ""
        while True:
            task = process_audio_task.AsyncResult(task_id)
            current_state = task.state
            current_info = task.info or {}
            payload = {'state': current_state, 'info': {}}
            if isinstance(current_info, dict): payload['info'] = current_info
            elif current_info is not None: payload['info']['status'] = f"Info: {str(current_info)}"
            # Refine status like in /status route
            if current_state == 'PENDING': payload['info'].setdefault('status', 'Waiting in queue...'); payload['info'].setdefault('percent', 0)
            elif current_state == 'STARTED': payload['info'].setdefault('status', 'Task started...'); payload['info'].setdefault('percent', 2)
            elif current_state == 'FAILURE': payload['info'].setdefault('status', f"Task failed: {str(current_info)}"); payload['info'].setdefault('percent', -1)
            elif current_state == 'SUCCESS': payload['info'].setdefault('status', 'Complete'); payload['info'].setdefault('percent', 100)

            try: current_data_str = json.dumps(payload, sort_keys=True)
            except TypeError: current_data_str = str(payload)

            if current_data_str != last_data_sent_str:
                logger.debug(f"SSE sending update for {task_id}: {payload}")
                yield f"data: {json.dumps(payload)}\n\n"
                last_data_sent_str = current_data_str

            if task.ready():
                logger.info(f"SSE stream closing for finished task {task_id} (State: {current_state})")
                break
            time.sleep(1)
    return Response(generate(), mimetype='text/event-stream')

@app.route("/download/<filename>")
def download_transcript(filename):
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
         logger.error(f"Download attempt with invalid filename format: {filename}")
         return "Invalid filename", 400
    transcript_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    if os.path.exists(transcript_path):
        logger.info(f"Providing transcript for download: {transcript_path}")
        try:
            # Attempt to create a slightly more user-friendly download name
            base_name = os.path.splitext(safe_filename)[0]
            if len(base_name) > 33 and base_name[32] == '_': # Check if it looks like UUID_original
                 download_name = f"transcript_{base_name[33:]}.txt"
            else:
                 download_name = f"transcript_{base_name}.txt" # Fallback
            return send_file(transcript_path, mimetype="text/plain", as_attachment=True, download_name=download_name)
        except Exception as send_err:
             logger.error(f"Error sending file {transcript_path}: {send_err}", exc_info=True)
             return "Error sending file.", 500
    else:
        logger.error(f"Transcript file not found for download: {transcript_path}")
        return "Transcript file not found. It may have been cleaned up or the task failed.", 404

# --------------------------------------------------------------------------
# Application Entry Point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Flask application on host 0.0.0.0 port {port}")
    # Use app.run for local development testing
    app.run(host='0.0.0.0', port=port, debug=False)
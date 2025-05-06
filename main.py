# --- Standard Library Imports ---
import os
import time
import logging
import uuid
import io
import json

# --- Third-party Library Imports ---
from flask import (
    Flask, request, jsonify, render_template, url_for, send_file, Response
)
from werkzeug.utils import secure_filename
from celery import Celery, current_task
from celery.utils.log import get_task_logger
import psutil
import torch
import whisperx
import soundfile as sf
import librosa
import numpy as np
from pyannote.audio import Pipeline
from pydub import AudioSegment
from tqdm import tqdm # Typically for CLI, but whisperx might use it internally
import ctranslate2 # Backend for whisperx
from dotenv import load_dotenv
import google.generativeai as genai

# --------------------------------------------------------------------------
# Load Environment Variables & Logging Configuration
# --------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__) # Global logger for Flask app parts

# --------------------------------------------------------------------------
# Constants and Configuration
# --------------------------------------------------------------------------
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "default_insecure_secret_key_please_change")
if FLASK_SECRET_KEY == "default_insecure_secret_key_please_change":
    logger.warning("FLASK_SECRET_KEY not set in environment. Using default (INSECURE). Set in .env file.")

UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024 # 2GB
AUDIO_CHUNK_DURATION_S = 300 # 5 minutes

# --- Gemini Configuration ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash-latest")

if GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info(f"Google Gemini API configured with model: {GEMINI_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to configure Google Gemini API: {e}")
        GOOGLE_API_KEY = None # Disable feature if config fails
else:
    logger.warning("GOOGLE_API_KEY not found in environment. Summarization feature will be disabled.")

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
# Ensure Redis is running: redis-server
app.config['broker_url'] = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.config['result_backend'] = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery = Celery(app.name, broker=app.config['broker_url'])
celery.conf.update(app.config)

# --------------------------------------------------------------------------
# HELPER FUNCTIONS (Keep existing functions as they are)
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
        logger.info(f"[{context}] CPU: {cpu_usage:.1f}%, Mem: {memory_usage:.1f}% ({memory_info.used/1024**3:.2f}/{memory_info.total/1024**3:.2f}GB)")
    except Exception as e:
        logger.warning(f"Could not log system usage: {e}")

def convert_to_wav(input_path, output_folder):
    try:
        file_root, _ = os.path.splitext(os.path.basename(input_path))
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
    num_audio_chunks = 0 # Renamed for clarity vs. loop counter
    try:
        logger.info(f"Loading audio for chunking: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None, mono=False) # Load with original sample rate, potentially stereo
        if y.ndim > 1 and y.shape[0] > 1: # If stereo
            y = librosa.to_mono(y)
            logger.info("Converted audio to mono for chunking.")

        total_duration_s = librosa.get_duration(y=y, sr=sr)
        if total_duration_s <= 0:
             logger.warning(f"Audio {audio_path} has zero or negative duration. Cannot chunk.")
             return [], 0

        chunk_size_samples = int(chunk_duration_s * sr)
        num_audio_chunks = int(np.ceil(total_duration_s / chunk_duration_s))

        logger.info(f"Chunking audio into ~{num_audio_chunks} chunks of {chunk_duration_s}s each.")
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]

        for i in range(num_audio_chunks):
            chunk_start_sample = i * chunk_size_samples
            chunk_end_sample = min((i + 1) * chunk_size_samples, len(y))
            chunk_audio_data = y[chunk_start_sample:chunk_end_sample]

            chunk_filename = os.path.join(output_folder, f"{base_filename}_chunk_{i:03d}.wav")
            logger.debug(f"Writing chunk {i+1}/{num_audio_chunks} to {chunk_filename}")
            sf.write(chunk_filename, chunk_audio_data, sr)
            chunks.append(chunk_filename)

        logger.info(f"Successfully created {len(chunks)} audio chunks.")
        return chunks, num_audio_chunks # Return actual number of chunks created
    except Exception as e:
        logger.error(f"Error during audio chunking for '{audio_path}': {e}", exc_info=True)
        for chunk_file in chunks: # Cleanup partially created chunks
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
# Model Loading Functions (Keep existing functions as they are)
# --------------------------------------------------------------------------
WHISPER_MODEL_CACHE = None
DIARIZATION_PIPELINE_CACHE = None

def load_whisper_model(device):
    global WHISPER_MODEL_CACHE
    if WHISPER_MODEL_CACHE:
        logger.info("Using cached Whisper model.")
        return WHISPER_MODEL_CACHE

    compute_type = "float16" if device == "cuda" else "int8"
    model = None
    try:
        logger.info(f"Loading Whisper large-v2 model with compute_type={compute_type} on device={device}...")
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)
        logger.info("Whisper model loaded successfully.")
    except (ValueError, RuntimeError, ctranslate2.translator.ModelleException if hasattr(ctranslate2, 'translator') and hasattr(ctranslate2.translator, 'ModelleException') else Exception) as e: # More robust exception handling for ctranslate2
        logger.warning(f"{compute_type} computation failed or not supported ({e}). Trying float32...")
        compute_type = "float32" # Fallback for CPU or older GPUs
        try:
            model = whisperx.load_model("large-v2", device, compute_type=compute_type)
            logger.info("Whisper model loaded successfully with compute_type=float32.")
        except Exception as e_fallback:
            logger.error(f"Failed to load Whisper model even with float32: {e_fallback}", exc_info=True)
            raise RuntimeError(f"Could not load Whisper model: {e_fallback}") from e_fallback
    except Exception as e_general:
        logger.error(f"An unexpected error occurred loading Whisper model: {e_general}", exc_info=True)
        raise RuntimeError(f"Could not load Whisper model: {e_general}") from e_general
    WHISPER_MODEL_CACHE = model
    return model

def load_diarization_pipeline(device):
    global DIARIZATION_PIPELINE_CACHE
    if DIARIZATION_PIPELINE_CACHE:
        logger.info("Using cached Diarization pipeline.")
        return DIARIZATION_PIPELINE_CACHE

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        logger.error("HUGGING_FACE_HUB_TOKEN env var not set. Cannot load diarization model.")
        raise ValueError("Missing HUGGING_FACE_HUB_TOKEN. Set in .env or environment.")
    pipeline = None
    try:
        logger.info("Loading Pyannote speaker-diarization-3.1 pipeline...") # Updated to 3.1
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
        pipeline.to(torch.device(device))
        logger.info("Diarization pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load diarization pipeline: {e}", exc_info=True)
        logger.error("Ensure you have accepted user agreements for pyannote models on Hugging Face Hub.")
        raise RuntimeError(f"Could not load diarization pipeline: {e}") from e
    DIARIZATION_PIPELINE_CACHE = pipeline
    return pipeline

# --------------------------------------------------------------------------
# Core Transcription and Diarization Logic (Keep existing functions as they are)
# --------------------------------------------------------------------------
def transcribe_and_diarize(audio_path, device, whisper_model, diarization_pipeline, progress_callback=None):
    # Uses global logger for its own messages, progress_callback uses task_logger
    log_system_usage(f"Start T&D: {os.path.basename(audio_path)}")
    if progress_callback: progress_callback({'status': 'Analyzing audio...', 'percent': 18})

    full_transcript_lines = []
    chunk_files_to_process = []
    created_chunk_files_to_cleanup = [] # Explicitly track created chunks for cleanup
    original_duration = 0
    num_expected_chunks = 1 # How many chunks we expect based on duration

    try:
        original_duration = librosa.get_duration(path=audio_path)
        logger.info(f"Original audio duration: {original_duration:.2f} seconds.")

        if original_duration > AUDIO_CHUNK_DURATION_S * 1.1: # If >10% longer than a single chunk
            logger.info("Audio is long, chunking...")
            if progress_callback: progress_callback({'status': 'Chunking audio...', 'percent': 19})
            # chunk_audio uses global logger
            created_chunks, num_expected_chunks = chunk_audio(audio_path, app.config['UPLOAD_FOLDER'], AUDIO_CHUNK_DURATION_S)
            if not created_chunks:
                logger.warning("Audio chunking resulted in no files, processing original.")
                chunk_files_to_process = [audio_path]
                num_expected_chunks = 1
            else:
                chunk_files_to_process = created_chunks
                created_chunk_files_to_cleanup = created_chunks # These need cleanup
        else:
            logger.info("Audio is short enough, processing as a single file.")
            chunk_files_to_process = [audio_path]

        if progress_callback: progress_callback({'status': 'Starting transcription & diarization loop...', 'percent': 20})

        total_offset = 0.0
        for i, chunk_path in enumerate(chunk_files_to_process):
            current_chunk_num = i + 1
            log_system_usage(f"Start Chunk {current_chunk_num}/{num_expected_chunks}")
            logger.info(f"Processing chunk {current_chunk_num}/{num_expected_chunks}: {os.path.basename(chunk_path)}")

            # Progress calculation for this chunk within the overall task progress
            base_task_percent = 18 # Percent where T&D starts
            total_task_span_for_T_and_D = 72 # 90 (end of T&D) - 18 (start of T&D)
            
            percent_start_chunk_in_task = base_task_percent + int(total_task_span_for_T_and_D * (i / num_expected_chunks))

            try:
                chunk_audio_data = whisperx.load_audio(chunk_path)
                # Estimate chunk duration if librosa fails (e.g., for a corrupt chunk)
                try: chunk_duration = librosa.get_duration(path=chunk_path)
                except Exception:
                    chunk_duration = len(chunk_audio_data) / 16000.0 if chunk_audio_data is not None and len(chunk_audio_data) > 0 else AUDIO_CHUNK_DURATION_S
                    logger.warning(f"Using estimated duration for chunk {os.path.basename(chunk_path)}")

                # 1. Transcription
                if progress_callback: progress_callback({'status': f'Transcribing chunk {current_chunk_num}/{num_expected_chunks}...', 'percent': percent_start_chunk_in_task + 5})
                result = whisper_model.transcribe(chunk_audio_data, batch_size=16) # batch_size is tunable
                if not result or "segments" not in result or not result["segments"]:
                    logger.warning(f"Whisper produced no segments for chunk: {os.path.basename(chunk_path)}")
                    total_offset += chunk_duration
                    if progress_callback:
                        percent_end_chunk_in_task = base_task_percent + int(total_task_span_for_T_and_D * ((i + 1) / num_expected_chunks))
                        progress_callback({'status': f'Chunk {current_chunk_num}/{num_expected_chunks} (no speech).', 'percent': percent_end_chunk_in_task})
                    continue

                # 2. Alignment
                if progress_callback: progress_callback({'status': f'Aligning chunk {current_chunk_num}/{num_expected_chunks}...', 'percent': percent_start_chunk_in_task + 10})
                aligned_result = result # Default to unaligned if alignment fails
                try:
                    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
                    aligned_result = whisperx.align(result["segments"], align_model, metadata, chunk_audio_data, device, return_char_alignments=False)
                    del align_model # Free memory
                    if device == 'cuda': torch.cuda.empty_cache()
                except Exception as align_error:
                    logger.warning(f"Alignment failed for chunk {os.path.basename(chunk_path)}: {align_error}. Using original Whisper timestamps.")

                # 3. Diarization
                if progress_callback: progress_callback({'status': f'Diarizing chunk {current_chunk_num}/{num_expected_chunks}...', 'percent': percent_start_chunk_in_task + 15})
                diarization_output = None
                try:
                    diarization_output = diarization_pipeline(chunk_path)
                except Exception as diarize_error: # pyannote can fail on very short/silent audio
                    logger.error(f"Diarization failed for chunk {os.path.basename(chunk_path)}: {diarize_error}", exc_info=False) # exc_info=False to reduce log verbosity for common issue

                # 4. Assign Speakers
                if progress_callback: progress_callback({'status': f'Assigning speakers chunk {current_chunk_num}/{num_expected_chunks}...', 'percent': percent_start_chunk_in_task + 20})
                for segment in aligned_result["segments"]:
                    seg_start, seg_end = segment.get("start"), segment.get("end")
                    seg_text = segment.get("text", "").strip()
                    if seg_start is None or seg_end is None or not seg_text: continue

                    global_start, global_end = total_offset + seg_start, total_offset + seg_end
                    dominant_speaker = "Unknown Speaker"
                    if diarization_output:
                        speaker_times = {}
                        try:
                            for turn, _, speaker_label in diarization_output.itertracks(yield_label=True):
                                overlap_start = max(turn.start, seg_start)
                                overlap_end = min(turn.end, seg_end)
                                overlap_duration = overlap_end - overlap_start
                                if overlap_duration > 0.01: # Min overlap to consider
                                    speaker_times[speaker_label] = speaker_times.get(speaker_label, 0) + overlap_duration
                            if speaker_times: dominant_speaker = max(speaker_times, key=speaker_times.get)
                        except Exception as speaker_assign_err:
                             logger.warning(f"Error assigning speaker: {speaker_assign_err}")
                    full_transcript_lines.append(f"[{global_start:08.2f}-{global_end:08.2f}] {dominant_speaker}: {seg_text}")

                total_offset += chunk_duration
                log_system_usage(f"End Chunk {current_chunk_num}/{num_expected_chunks}")
                if progress_callback:
                    percent_end_chunk_in_task = base_task_percent + int(total_task_span_for_T_and_D * ((i + 1) / num_expected_chunks))
                    progress_callback({'status': f'Chunk {current_chunk_num}/{num_expected_chunks} processed.', 'percent': percent_end_chunk_in_task})

            except Exception as chunk_proc_error:
                 logger.error(f"Failed to process chunk {os.path.basename(chunk_path)}: {chunk_proc_error}", exc_info=True)
                 if progress_callback:
                     percent_end_chunk_in_task = base_task_percent + int(total_task_span_for_T_and_D * ((i + 1) / num_expected_chunks))
                     progress_callback({'status': f'Error on chunk {current_chunk_num}/{num_expected_chunks}.', 'percent': percent_end_chunk_in_task })
                 try: # Attempt to advance offset even if chunk failed
                     failed_chunk_dur = librosa.get_duration(path=chunk_path)
                     total_offset += failed_chunk_dur
                 except: total_offset += AUDIO_CHUNK_DURATION_S # Fallback

    except Exception as e:
        logger.error(f"Fatal error during T&D for {audio_path}: {e}", exc_info=True)
        return {"transcript": f"[Processing Error: {str(e)}]", "num_speakers": 0, "audio_duration": original_duration or 0.1}
    finally:
        if created_chunk_files_to_cleanup:
            logger.info("Cleaning up temporary audio chunks...")
            safe_cleanup(*created_chunk_files_to_cleanup)
        log_system_usage(f"End T&D: {os.path.basename(audio_path)}")

    full_transcript_text = "\n".join(full_transcript_lines)
    if not full_transcript_lines:
         logger.warning(f"T&D completed but no transcript generated for {audio_path}.")
         if progress_callback: progress_callback({'status': 'Processing complete (no speech).', 'percent': 90})
         return {"transcript": "[No speech detected or processing error resulted in empty transcript]", "num_speakers": 0, "audio_duration": original_duration}

    # Estimate number of unique speakers from transcript
    all_speaker_labels = set(line.split(']', 1)[1].split(':', 1)[0].strip() for line in full_transcript_lines if ']' in line and ':' in line.split(']',1)[1])
    all_speaker_labels.discard("Unknown Speaker") # Don't count "Unknown Speaker" as a unique speaker
    estimated_num_speakers = len(all_speaker_labels) if all_speaker_labels else (1 if full_transcript_text.strip() else 0)

    logger.info(f"Estimated number of unique identified speakers: {estimated_num_speakers}")
    if progress_callback: progress_callback({'status': 'Transcription & Diarization complete.', 'percent': 90})
    return {"transcript": full_transcript_text, "num_speakers": estimated_num_speakers, "audio_duration": original_duration}

# --------------------------------------------------------------------------
# Gemini Summarization Function (Keep existing functions as they are)
# --------------------------------------------------------------------------
def generate_summary_with_gemini(transcript_text, audio_duration_seconds, num_speakers_detected):
    task_logger_instance = get_task_logger(__name__) # Use Celery's task logger

    if not GOOGLE_API_KEY:
        task_logger_instance.warning("Gemini API key not available. Skipping summarization.")
        return "Summarization disabled: API key not configured."
    if not GEMINI_MODEL_NAME:
        task_logger_instance.warning("GEMINI_MODEL not set. Skipping summarization.")
        return "Summarization disabled: Model not configured."

    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    except Exception as e:
        task_logger_instance.error(f"Failed to initialize Gemini model '{GEMINI_MODEL_NAME}': {e}")
        return f"Summarization failed: Could not load model ({str(e)})."

    minutes = int(audio_duration_seconds // 60)
    seconds = int(audio_duration_seconds % 60)
    audio_duration_formatted = f"{minutes} minute(s) and {seconds} second(s)"

    # Truncate transcript for prompt if it's excessively long to avoid hitting API limits hard, though Gemini 1.5 Flash has large context
    # This is a soft limit for the prompt example.
    max_prompt_transcript_len = 1000000 # Approx 1 million characters
    truncated_transcript_info = ""
    if len(transcript_text) > max_prompt_transcript_len:
        transcript_for_prompt = transcript_text[:max_prompt_transcript_len]
        truncated_transcript_info = f"\n... (transcript truncated for this prompt; full length: {len(transcript_text)} chars)"
    else:
        transcript_for_prompt = transcript_text


    prompt = f"""Analyze the following audio transcript and generate a structured summary.

Audio Transcript:
\"\"\"
{transcript_for_prompt}
\"\"\"{truncated_transcript_info}

Contextual Information from System:
- Audio Length: {audio_duration_formatted}
- Number of Unique Speakers Identified in Transcript: {num_speakers_detected} (Note: This excludes 'Unknown Speaker' labels)

Please generate a summary document with the following structure.
For each category, provide specific statements or a concise summary based *only* on the provided transcript.
If no information is found for a category in the transcript, explicitly state "None identified."

--- SUMMARY DOCUMENT ---
Number of speakers: {num_speakers_detected}
Length of audio: {audio_duration_formatted}

Statements about money:
[Extract key quotes or summarize discussions related to finance, currency, cost, wealth, poverty, etc. If none, state "None identified."]

Statements about drugs:
[Extract key quotes or summarize discussions related to illicit substances, medication, addiction, etc. If none, state "None identified."]

Statements about violence:
[Extract key quotes or summarize discussions related to physical harm, aggression, weapons, conflict, etc. If none, state "None identified."]

Statements about sex or sexuality:
[Extract key quotes or summarize discussions related to sexual acts, gender identity, sexual orientation, relationships, intimacy etc. If none, state "None identified."]

Key themes:
[List 3-5 predominant themes or topics discussed in the audio. Be concise.]
--- END SUMMARY DOCUMENT ---

Ensure your entire response adheres to the format above, starting with "Number of speakers:" and ending after "Key themes:".
Do not add any introductory or concluding remarks outside of this structure.
"""
    try:
        task_logger_instance.info(f"Sending prompt to Gemini model {GEMINI_MODEL_NAME}. Transcript for prompt length: {len(transcript_for_prompt)} chars.")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=0.3),
        )
        
        if not response.candidates or not response.candidates[0].content.parts:
            error_message = "Gemini response was empty."
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                reason = response.prompt_feedback.block_reason.name
                error_message += f" Blocked. Reason: {reason}"
                task_logger_instance.error(f"Gemini content generation blocked. Reason: {reason} - Safety Ratings: {response.prompt_feedback.safety_ratings}")
            else:
                # Accessing parts might be different based on API version or response structure
                response_parts_str = str(response.parts) if hasattr(response, 'parts') else "N/A"
                response_candidates_str = str(response.candidates) if hasattr(response, 'candidates') else "N/A"
                task_logger_instance.error(error_message + f" - Full Response (parts): {response_parts_str} (candidates): {response_candidates_str}")
            return f"Summarization failed: {error_message}"

        summary_text = response.text # .text should directly access the combined text from parts
        task_logger_instance.info("Successfully received summary from Gemini.")
        return summary_text

    except Exception as e:
        task_logger_instance.error(f"Error during Gemini API call: {e}", exc_info=True)
        return f"Summarization failed due to an API error: {str(e)}"

# --------------------------------------------------------------------------
# CELERY TASK DEFINITION (Keep existing functions as they are)
# --------------------------------------------------------------------------
@celery.task(bind=True)
def process_audio_task(self, temp_audio_path, original_filename):
    task_logger = get_task_logger(__name__) # Task-specific logger
    task_id = self.request.id
    task_logger.info(f"Task {task_id}: Starting processing for {original_filename}")

    transcript_file_uuid = uuid.uuid4().hex
    transcript_filename_base = f"{transcript_file_uuid}_transcript.txt"
    upload_dir = app.config['UPLOAD_FOLDER']
    transcript_path = os.path.join(upload_dir, transcript_filename_base)
    
    processed_audio_path_for_transcription = temp_audio_path
    original_uploaded_temp_path = temp_audio_path

    def progress_callback(update_data):
        task_logger.info(f"Task {task_id} Progress: {update_data}")
        try:
            update_data['task_id'] = task_id
            self.update_state(state='PROGRESS', meta=update_data)
        except Exception as e:
            task_logger.error(f"Task {task_id}: Failed to update progress state: {e}")

    try:
        progress_callback({'status': 'Preparing...', 'percent': 5})
        log_system_usage(f"Task {task_id} Prepare")

        if not original_filename.lower().endswith(".wav"):
             progress_callback({'status': 'Converting audio to WAV...', 'percent': 10})
             processed_audio_path_for_transcription = convert_to_wav(temp_audio_path, upload_dir)
             progress_callback({'status': 'Conversion complete.', 'percent': 15})

        progress_callback({'status': 'Loading AI models...', 'percent': 16})
        device = get_cuda_device()
        whisper_model_instance = load_whisper_model(device)
        diarization_pipeline_instance = load_diarization_pipeline(device)
        progress_callback({'status': 'AI Models ready.', 'percent': 17})

        transcription_data = transcribe_and_diarize(
            processed_audio_path_for_transcription,
            device,
            whisper_model_instance,
            diarization_pipeline_instance,
            progress_callback=progress_callback
        )
        transcript_content = transcription_data["transcript"]
        num_speakers = transcription_data["num_speakers"]
        audio_duration = transcription_data["audio_duration"]

        progress_callback({'status': 'Saving transcript...', 'percent': 91})
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_content)
        task_logger.info(f"Task {task_id}: Transcript saved to: {transcript_path}")

        summary_content = "Summarization was not attempted or an error occurred."
        summary_filename_base = None
        if GOOGLE_API_KEY and GEMINI_MODEL_NAME:
            if "[Processing Error:" in transcript_content or "[No speech detected" in transcript_content:
                task_logger.warning(f"Task {task_id}: Skipping summarization due to problematic transcript.")
                summary_content = "Summarization skipped: Transcript indicates processing error or no speech."
            else:
                progress_callback({'status': 'Generating summary with Gemini AI...', 'percent': 93})
                summary_content = generate_summary_with_gemini(
                    transcript_text=transcript_content,
                    audio_duration_seconds=audio_duration,
                    num_speakers_detected=num_speakers
                )
                if not summary_content.startswith("Summarization failed") and not summary_content.startswith("Summarization disabled"):
                    summary_file_uuid = uuid.uuid4().hex
                    summary_filename_base = f"{summary_file_uuid}_summary.txt"
                    summary_file_path = os.path.join(upload_dir, summary_filename_base)
                    with open(summary_file_path, 'w', encoding='utf-8') as f:
                        f.write(summary_content)
                    task_logger.info(f"Task {task_id}: Summary saved to: {summary_file_path}")
                else:
                    task_logger.warning(f"Task {task_id}: Summarization was not successful. Content: {summary_content[:200]}...")
        else:
            task_logger.warning(f"Task {task_id}: Gemini API Key or Model not set. Skipping summarization.")
            summary_content = "Summarization disabled: API Key or Model not configured."
        
        progress_callback({'status': 'Finalizing results...', 'percent': 98})

        original_base_filename_for_download = os.path.splitext(secure_filename(original_filename))[0]
        final_meta = {
            'status': 'Complete', 'percent': 100,
            'result_filename': transcript_filename_base,
            'summary_filename': summary_filename_base,
            'summary_content': summary_content,
            'original_base_filename': original_base_filename_for_download
        }
        log_system_usage(f"Task {task_id} Complete")
        return final_meta

    except Exception as e:
        error_message = f"Critical error in task for {original_filename}: {str(e)}"
        task_logger.error(f"Task {task_id}: {error_message}", exc_info=True)
        log_system_usage(f"Task {task_id} Failed")
        self.update_state(
            state='FAILURE',
            meta={'status': error_message, 'percent': -1, 'exc_type': type(e).__name__, 'exc_message': str(e)}
        )
    finally:
        task_logger.info(f"Task {task_id}: Running cleanup.")
        safe_cleanup(original_uploaded_temp_path)
        if processed_audio_path_for_transcription and processed_audio_path_for_transcription != original_uploaded_temp_path:
            safe_cleanup(processed_audio_path_for_transcription)
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            task_logger.info(f"Task {task_id}: Cleared CUDA cache (if applicable).")

# --------------------------------------------------------------------------
# Flask Routes (Keep existing functions as they are)
# --------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index_route():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_audio():
    if "audio_file" not in request.files:
        logger.warning("Upload: No file part.")
        return jsonify({"error": "No file part selected."}), 400
    audio_file = request.files["audio_file"]
    if audio_file.filename == "":
        logger.warning("Upload: No filename.")
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(audio_file.filename):
        logger.warning(f"Upload: Invalid file type: {audio_file.filename}")
        return jsonify({"error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    filename = secure_filename(audio_file.filename)
    unique_id = uuid.uuid4().hex
    temp_audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")

    try:
        logger.info(f"Receiving file: {filename}")
        audio_file.save(temp_audio_path)
        logger.info(f"File saved temporarily to: {temp_audio_path}")
        task = process_audio_task.delay(temp_audio_path, filename)
        logger.info(f"Delegated processing for {filename} to Celery task ID: {task.id}")
        return jsonify({"task_id": task.id}), 202 # 202 Accepted
    except Exception as e:
        logger.error(f"Error during upload processing for {filename}: {e}", exc_info=True)
        safe_cleanup(temp_audio_path)
        return jsonify({"error": f"Server error during upload: {str(e)}"}), 500

def _get_task_response_data(task_id):
    task = process_audio_task.AsyncResult(task_id)
    response_data = {'state': task.state, 'info': {}}
    default_info = {'status': 'Initializing...', 'percent': 0, 'result_filename': None,
                    'summary_filename': None, 'summary_content': None, 'original_base_filename': None}
    if isinstance(task.info, dict):
        response_data['info'] = {**default_info, **task.info}
    elif task.info is not None:
        response_data['info'] = {**default_info, 'status': f"Error: {str(task.info)}"}
    else:
        response_data['info'] = default_info

    if task.state == 'PENDING':
        response_data['info']['status'] = response_data['info'].get('status', 'Waiting in queue...')
        response_data['info']['percent'] = response_data['info'].get('percent', 0)
    elif task.state == 'STARTED':
        response_data['info']['status'] = response_data['info'].get('status', 'Task processing started...')
        response_data['info']['percent'] = response_data['info'].get('percent', 2)
    elif task.state == 'RETRY':
        response_data['info']['status'] = f"Task retrying: {str(task.info or 'details unavailable')}"
        response_data['info']['percent'] = 0
    elif task.state == 'FAILURE':
        if not response_data['info'].get('status', '').lower().startswith(('error', 'task failed', 'critical error')):
             response_data['info']['status'] = f"Task failed: {str(task.info or 'details unavailable')}"
        response_data['info']['percent'] = -1
    elif task.state == 'SUCCESS':
        response_data['info']['status'] = response_data['info'].get('status', 'Complete')
        response_data['info']['percent'] = 100
    return response_data, task.ready()

@app.route("/status/<task_id>")
def task_status(task_id):
    response_data, _ = _get_task_response_data(task_id)
    return jsonify(response_data)

@app.route('/stream-progress/<task_id>')
def stream_progress(task_id):
    def generate():
        last_data_sent_str = ""
        while True:
            response_data, task_ready = _get_task_response_data(task_id)
            try:
                current_data_str = json.dumps(response_data, sort_keys=True)
            except TypeError:
                current_data_str = str(response_data)
            if current_data_str != last_data_sent_str:
                logger.debug(f"SSE sending update for {task_id}: {response_data}")
                yield f"data: {json.dumps(response_data)}\n\n"
                last_data_sent_str = current_data_str
            if task_ready:
                logger.info(f"SSE stream closing for finished task {task_id} (State: {response_data['state']})")
                break
            time.sleep(1)
    return Response(generate(), mimetype='text/event-stream')

@app.route("/download/<filename>")
def download_transcript(filename):
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
         logger.error(f"Download: Invalid filename format: {filename}")
         return "Invalid filename", 400
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    if os.path.exists(file_path):
        logger.info(f"Providing file for download: {file_path}")
        sensible_download_name = filename 
        try:
            return send_file(file_path, mimetype="text/plain", as_attachment=True, download_name=sensible_download_name)
        except Exception as send_err:
             logger.error(f"Error sending file {file_path}: {send_err}", exc_info=True)
             return "Error sending file.", 500
    else:
        logger.error(f"File not found for download: {file_path}")
        return "File not found. It may have been cleaned up or the task failed.", 404

# --------------------------------------------------------------------------
# Application Entry Point --- MODIFIED FOR GEVENT ---
# --------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # FLASK_DEBUG env var controls which server is used.
    # 'true' (case-insensitive) uses Flask dev server.
    # Otherwise (e.g., 'false', '0', or not set) uses gevent.
    debug_env_value = os.environ.get("FLASK_DEBUG", "false").lower()
    use_flask_dev_server = debug_env_value == "true" or debug_env_value == "1"

    if use_flask_dev_server:
        logger.info(f"Starting Flask development server on host 0.0.0.0 port {port}, Debug: True")
        # Flask's built-in server is fine for debugging, not for production
        # It also handles reloading on code changes well.
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        # Use gevent for production or when FLASK_DEBUG is not explicitly true
        try:
            from gevent.pywsgi import WSGIServer
            logger.info(f"Starting gevent WSGI server on host 0.0.0.0 port {port}")
            http_server = WSGIServer(('0.0.0.0', port), app)
            http_server.serve_forever()
        except ImportError:
            logger.error("gevent not found. Please ensure it's installed. Falling back to Flask dev server (not recommended for production).")
            app.run(host='0.0.0.0', port=port, debug=False) # Fallback, not ideal for prod
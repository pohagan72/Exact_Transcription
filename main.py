# --- Standard Library Imports ---
import os
import time
import logging
import uuid
import io  # Keep io if needed elsewhere, but pydub/librosa handle most file ops

# --- Third-party Library Imports ---
import psutil         # For system resource monitoring
import torch          # PyTorch core
import whisperx       # WhisperX for transcription and alignment
import soundfile as sf # For writing audio files (used by chunk_audio example)
import librosa        # For audio loading and duration calculation
import numpy as np    # For numerical operations (often used by audio libs)
from flask import (   # Flask web framework components
    Flask, render_template, request, redirect, url_for,
    flash, send_file, session
)
from werkzeug.utils import secure_filename # For securing uploaded filenames
from pyannote.audio import Pipeline        # For speaker diarization
from pydub import AudioSegment             # For audio format conversion
from tqdm import tqdm                      # For progress bars (optional but nice)
import ctranslate2    # Dependency for whisperx/faster-whisper
from dotenv import load_dotenv             # For loading .env files

# --------------------------------------------------------------------------
# Load Environment Variables
# --------------------------------------------------------------------------
# Load variables from .env file into environment variables
# This should be called early, before accessing os.environ variables below
load_dotenv()
logger = logging.getLogger(__name__) # Get logger after potential env var config

# --------------------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------------------
# Configure logging level and format
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# --------------------------------------------------------------------------
# Constants and Configuration
# --------------------------------------------------------------------------
# Use environment variables or defaults for configuration
FLASK_SECRET_KEY = os.environ.get("FLASK_SECRET_KEY")
if not FLASK_SECRET_KEY:
    logger.warning("FLASK_SECRET_KEY not set in environment. Using default (INSECURE). Set in .env file.")
    FLASK_SECRET_KEY = "default_insecure_secret_key_please_change"

# Ensure the upload folder exists
UPLOAD_FOLDER = 'temp_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define allowed audio file extensions
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'}

# Maximum file size (e.g., 2 GB)
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024

# Chunk duration for processing large files (in seconds)
AUDIO_CHUNK_DURATION_S = 300 # 5 minutes


# --------------------------------------------------------------------------
# Flask App Initialization
# --------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_BYTES


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------
def get_cuda_device():
    """Detects if CUDA is available and returns the appropriate device string."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA is available. Using GPU.")
    else:
        device = "cpu"
        logger.info("CUDA is not available. Using CPU.")
    return device

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def log_system_usage(context="General"):
    """Logs current CPU and Memory usage."""
    try:
        cpu_usage = psutil.cpu_percent(interval=None) # Non-blocking
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent
        logger.info(f"[{context}] CPU Usage: {cpu_usage:.1f}%, Memory Usage: {memory_usage:.1f}% ({memory_info.used/1024**3:.2f}/{memory_info.total/1024**3:.2f} GB)")
    except Exception as e:
        logger.warning(f"Could not log system usage: {e}")

def convert_to_wav(input_path, output_folder):
    """Converts an audio file to WAV format using pydub."""
    try:
        file_root, file_ext = os.path.splitext(os.path.basename(input_path))
        # Ensure unique output filename if needed, or just use base name
        wav_filename = f"{file_root}_converted.wav"
        wav_path = os.path.join(output_folder, wav_filename)

        logger.info(f"Attempting to convert {input_path} to WAV...")
        audio = AudioSegment.from_file(input_path)
        # Export with parameters suitable for Whisper/Pyannote if needed (e.g., mono, 16kHz)
        # audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(wav_path, format="wav")
        logger.info(f"Successfully converted to WAV: {wav_path}")
        return wav_path
    except Exception as e:
        logger.error(f"Error converting file '{input_path}' to WAV: {e}", exc_info=True)
        raise RuntimeError(f"Failed to convert file to WAV: {e}") from e

def chunk_audio(audio_path, output_folder, chunk_duration_s=AUDIO_CHUNK_DURATION_S):
    """Splits an audio file into smaller chunks."""
    chunks = []
    try:
        logger.info(f"Loading audio for chunking: {audio_path}")
        y, sr = librosa.load(audio_path, sr=None, mono=False) # Load with original sample rate and channels

        if y.ndim > 1 and y.shape[0] > 1: # If stereo or more channels, convert to mono for chunking/processing
            y = librosa.to_mono(y)
            logger.info("Converted audio to mono for chunking.")

        total_duration_s = librosa.get_duration(y=y, sr=sr)
        chunk_size_samples = int(chunk_duration_s * sr)
        num_chunks = int(np.ceil(total_duration_s / chunk_duration_s))

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
        return chunks
    except Exception as e:
        logger.error(f"Error during audio chunking for '{audio_path}': {e}", exc_info=True)
        # Clean up any partially created chunks if error occurs
        for chunk_file in chunks:
            if os.path.exists(chunk_file):
                try:
                    os.remove(chunk_file)
                except OSError:
                    logger.warning(f"Could not clean up chunk file: {chunk_file}")
        raise RuntimeError(f"Failed to chunk audio: {e}") from e

def safe_cleanup(*filepaths):
    """Attempts to remove files, logging any errors."""
    for filepath in filepaths:
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
                logger.info(f"Cleaned up temporary file: {filepath}")
            except OSError as e:
                logger.warning(f"Could not remove temporary file {filepath}: {e}")


# --------------------------------------------------------------------------
# Model Loading Functions (Consider caching or global context if possible)
# --------------------------------------------------------------------------
def load_whisper_model(device):
    """Loads the WhisperX model, trying float16 first."""
    compute_type = "float16" if device == "cuda" else "int8" # Or float32 for CPU if int8 causes issues
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
    """Loads the Pyannote diarization pipeline using token from environment."""
    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        logger.error("HUGGING_FACE_HUB_TOKEN environment variable not set. Cannot load diarization model.")
        raise ValueError("Missing HUGGING_FACE_HUB_TOKEN. Please set it in the .env file or environment.")

    pipeline = None
    try:
        logger.info("Loading Pyannote speaker-diarization-3.0 pipeline...")
        # Ensure use_auth_token is used correctly. For newer pyannote it might be automatic if logged in via CLI or huggingface_hub.login()
        # Passing the token directly is usually reliable.
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=hf_token
        )
        pipeline.to(torch.device(device))
        logger.info("Diarization pipeline loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load diarization pipeline: {e}", exc_info=True)
        logger.error("Ensure you have accepted the user agreement for pyannote/speaker-diarization-3.0 and segmentation-3.0 on Hugging Face Hub.")
        raise RuntimeError(f"Could not load diarization pipeline: {e}") from e

    return pipeline


# --------------------------------------------------------------------------
# Core Transcription and Diarization Logic
# --------------------------------------------------------------------------
def transcribe_and_diarize(audio_path, device, whisper_model, diarization_pipeline):
    """Performs transcription, alignment, and diarization on an audio file."""
    log_system_usage(f"Start Processing: {os.path.basename(audio_path)}")
    full_transcript_lines = []
    chunk_files = []
    original_duration = 0

    try:
        original_duration = librosa.get_duration(path=audio_path)
        logger.info(f"Original audio duration: {original_duration:.2f} seconds.")

        # Decide whether to chunk based on duration
        if original_duration > AUDIO_CHUNK_DURATION_S * 1.1: # Add a buffer
            logger.info("Audio is long, chunking...")
            chunk_files = chunk_audio(audio_path, UPLOAD_FOLDER)
            if not chunk_files:
                 raise RuntimeError("Audio chunking resulted in no files.")
        else:
            logger.info("Audio is short enough, processing as a single file.")
            chunk_files = [audio_path] # Treat the whole file as a single 'chunk'

        total_offset = 0.0 # Keep track of time offset for combining chunks
        cumulative_processed_duration = 0.0

        for i, chunk_path in enumerate(tqdm(chunk_files, desc="Processing Audio Chunks", unit="chunk")):
            logger.info(f"Processing chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_path)}")
            log_system_usage(f"Start Chunk {i+1}")

            try:
                chunk_audio_data = whisperx.load_audio(chunk_path)
                chunk_duration = len(chunk_audio_data) / 16000 # Assuming 16kHz needed by whisperx

                # 1. Transcription
                logger.debug("Running Whisper transcription...")
                result = whisper_model.transcribe(chunk_audio_data, batch_size=16) # Adjust batch_size based on VRAM
                if not result or "segments" not in result or not result["segments"]:
                    logger.warning(f"Whisper produced no segments for chunk: {os.path.basename(chunk_path)}")
                    total_offset += chunk_duration # Update offset even if no transcript
                    cumulative_processed_duration += chunk_duration
                    continue # Skip to the next chunk

                logger.debug(f"Transcription found {len(result['segments'])} segments.")

                # 2. Alignment
                try:
                    logger.debug("Loading alignment model...")
                    align_model, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
                    logger.debug("Running alignment...")
                    aligned_result = whisperx.align(result["segments"], align_model, metadata, chunk_audio_data, device, return_char_alignments=False)
                    del align_model # Free memory
                    if device == 'cuda': torch.cuda.empty_cache()
                    logger.debug("Alignment complete.")
                except Exception as align_error:
                    logger.warning(f"Alignment failed for chunk {os.path.basename(chunk_path)}: {align_error}. Using original Whisper timestamps.")
                    aligned_result = result # Fallback

                # 3. Diarization
                try:
                    logger.debug("Running speaker diarization...")
                    diarization = diarization_pipeline(chunk_path) # Pyannote works on file paths
                    logger.debug("Diarization complete.")
                except Exception as diarize_error:
                    logger.error(f"Diarization failed for chunk {os.path.basename(chunk_path)}: {diarize_error}", exc_info=True)
                    diarization = None # Proceed without speaker labels for this chunk

                # 4. Assign Speakers to Segments
                for segment in aligned_result["segments"]:
                    seg_start = segment.get("start")
                    seg_end = segment.get("end")
                    seg_text = segment.get("text", "").strip()

                    if seg_start is None or seg_end is None or not seg_text:
                        logger.debug(f"Skipping segment with missing start/end/text: {segment}")
                        continue # Skip segments without clear timing or text

                    # Adjust segment times by the offset of previous chunks
                    global_start = total_offset + seg_start
                    global_end = total_offset + seg_end

                    dominant_speaker = "Unknown Speaker"
                    if diarization:
                        speaker_times = {}
                        try:
                            for turn, _, speaker in diarization.itertracks(yield_label=True):
                                # Calculate overlap between diarization turn and Whisper segment (within the chunk's timeframe)
                                overlap_start = max(turn.start, seg_start)
                                overlap_end = min(turn.end, seg_end)
                                overlap_duration = overlap_end - overlap_start

                                if overlap_duration > 0:
                                    speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap_duration

                            if speaker_times:
                                dominant_speaker = max(speaker_times, key=speaker_times.get)
                        except Exception as speaker_assign_err:
                             logger.warning(f"Error assigning speaker to segment [{seg_start:.2f}-{seg_end:.2f}]: {speaker_assign_err}")


                    formatted_line = f"[{global_start:08.2f}-{global_end:08.2f}] {dominant_speaker}: {seg_text}"
                    full_transcript_lines.append(formatted_line)
                    logger.debug(f"Added line: {formatted_line}")

                # Update the time offset for the next chunk
                total_offset += chunk_duration
                cumulative_processed_duration += chunk_duration
                logger.debug(f"Cumulative processed duration: {cumulative_processed_duration:.2f}s")
                log_system_usage(f"End Chunk {i+1}")

            except Exception as chunk_proc_error:
                 logger.error(f"Failed to process chunk {os.path.basename(chunk_path)}: {chunk_proc_error}", exc_info=True)
                 # Decide if you want to stop or continue with next chunk
                 flash(f"Error processing part of the audio: {chunk_proc_error}. Result may be incomplete.", "warning")
                 # Update offset based on estimated chunk duration if possible
                 try:
                     chunk_dur = librosa.get_duration(path=chunk_path)
                     total_offset += chunk_dur
                     cumulative_processed_duration += chunk_dur
                 except Exception:
                     logger.warning(f"Could not get duration for failed chunk {chunk_path}. Offset might be inaccurate.")
                     total_offset += AUDIO_CHUNK_DURATION_S # Estimate

    except Exception as e:
        logger.error(f"Fatal error during transcription/diarization process for {audio_path}: {e}", exc_info=True)
        raise RuntimeError(f"Processing failed: {e}") from e
    finally:
        # Clean up chunk files only if they were created (i.e., not the original input)
        if chunk_files and len(chunk_files) > 1 or (chunk_files and chunk_files[0] != audio_path):
            logger.info("Cleaning up temporary audio chunks...")
            safe_cleanup(*chunk_files)

        log_system_usage(f"End Processing: {os.path.basename(audio_path)}")

    if not full_transcript_lines:
         logger.warning(f"Processing completed but no transcript was generated for {audio_path}.")
         return "[No speech detected or processing error resulted in empty transcript]"

    return "\n".join(full_transcript_lines)


# --------------------------------------------------------------------------
# Flask Routes
# --------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    """Handles file upload and initiates the transcription process."""
    if request.method == "POST":
        # --- File Validation ---
        if "audio_file" not in request.files:
            flash("No file part selected.", "error")
            return redirect(request.url)

        audio_file = request.files["audio_file"]
        if audio_file.filename == "":
            flash("No file selected.", "error")
            return redirect(request.url)

        if not allowed_file(audio_file.filename):
            flash(f"Invalid file type. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}", "error")
            return redirect(request.url)

        # --- File Saving ---
        filename = secure_filename(audio_file.filename)
        # Use UUID to prevent filename collisions and make temporary files unique
        unique_id = str(uuid.uuid4())
        temp_audio_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{filename}")
        processed_audio_path = None # Path after potential conversion
        transcript_filename = f"{unique_id}_transcript.txt"
        transcript_path = os.path.join(UPLOAD_FOLDER, transcript_filename)

        try:
            logger.info(f"Receiving file: {filename}")
            audio_file.save(temp_audio_path)
            logger.info(f"File saved temporarily to: {temp_audio_path}")

            # Get duration for user feedback
            try:
                duration = librosa.get_duration(path=temp_audio_path)
                duration_minutes = int(duration // 60)
                duration_seconds = int(duration % 60)
                flash(f"File '{filename}' ({duration_minutes}m {duration_seconds}s) received. Starting processing...", "info")
                logger.info(f"Audio duration: {duration:.2f} seconds")
            except Exception as dur_err:
                logger.warning(f"Could not get duration for {temp_audio_path}: {dur_err}")
                flash(f"File '{filename}' received. Processing...", "info") # Inform user anyway

            # --- Processing ---
            device = get_cuda_device()
            whisper_model = load_whisper_model(device) # Load models per request (can be slow)
            diarization_pipeline = load_diarization_pipeline(device) # Reads token from env

            # Convert to WAV if necessary (required by some tools)
            processed_audio_path = temp_audio_path
            if not filename.lower().endswith(".wav"):
                 processed_audio_path = convert_to_wav(temp_audio_path, UPLOAD_FOLDER)

            # Run the main processing function
            logger.info("Starting transcription and diarization...")
            start_time = time.time()
            transcript_content = transcribe_and_diarize(processed_audio_path, device, whisper_model, diarization_pipeline)
            end_time = time.time()
            logger.info(f"Transcription and diarization finished in {end_time - start_time:.2f} seconds.")

            # --- Save Transcript ---
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript_content)
            logger.info(f"Transcript saved to: {transcript_path}")

            # Store filename in session for download route
            session['transcript_filename'] = transcript_filename
            flash("Transcription complete! Click below to download.", "success")
            return redirect(url_for('download_transcript')) # Redirect to download page/button

        except (ValueError, RuntimeError) as e: # Catch specific errors we raise
            flash(f"Processing Error: {str(e)}", "danger") # Use danger for critical errors
            logger.error(f"Error during processing request for {filename}: {e}", exc_info=True)
            # Clean up session key if error occurred before redirect
            session.pop('transcript_filename', None)
            # No redirect here, will fall through to render index again with flash message

        except Exception as e:
            flash(f"An unexpected error occurred. Please try again later.", "danger")
            logger.error(f"Unexpected error processing {filename}: {e}", exc_info=True)
            session.pop('transcript_filename', None)
            # No redirect here

        finally:
            # --- Cleanup ---
            # Clean up the original uploaded file and the potentially converted WAV file
            safe_cleanup(temp_audio_path)
            if processed_audio_path and processed_audio_path != temp_audio_path:
                safe_cleanup(processed_audio_path)
            # Don't clean up transcript file here; download route needs it. It should be cleaned *after* download or via a separate job.

    # Render the main page on GET request or after a POST error that didn't redirect
    return render_template("index.html")

@app.route("/download")
def download_transcript():
    """Provides the generated transcript file for download."""
    transcript_filename = session.get('transcript_filename', None)
    if not transcript_filename:
        flash("No transcript available for download, or your session expired. Please upload again.", "warning")
        return redirect(url_for('index'))

    # Prevent path traversal - ensure filename is just a filename
    transcript_filename = secure_filename(transcript_filename)
    transcript_path = os.path.join(UPLOAD_FOLDER, transcript_filename)

    if os.path.exists(transcript_path):
        logger.info(f"Providing transcript for download: {transcript_path}")
        # Pop the key AFTER confirming file exists and BEFORE sending
        # to prevent repeated downloads from the same session state if something goes wrong during send_file
        session.pop('transcript_filename', None)
        try:
            # Return the file to the user
            return send_file(
                transcript_path,
                mimetype="text/plain",
                as_attachment=True,
                download_name="transcript.txt" # User-friendly download name
                # Consider adding: conditional=True for ETag/caching support
            )
        finally:
            # Optional: Clean up the transcript file *after* attempting to send it.
            # Be cautious: if the download fails mid-way, the file is gone.
            # A safer approach is a background job to clean up old files.
            # safe_cleanup(transcript_path)
            pass # Keep file for now unless cleanup is explicitly required
    else:
        flash("Transcript file could not be found on the server. It might have been cleaned up.", "error")
        logger.error(f"Transcript file not found at expected path for download: {transcript_path}")
        session.pop('transcript_filename', None) # Clear invalid session key
        return redirect(url_for('index'))

# --------------------------------------------------------------------------
# Application Entry Point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Get port from environment variable (used by Cloud Run, etc.) or default
    port = int(os.environ.get("PORT", 5000)) # Default Flask port
    # Run the app
    # Set debug=False for production environments
    # host='0.0.0.0' makes it accessible from outside the container/machine
    logger.info(f"Starting Flask application on host 0.0.0.0 port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) # Set debug=True ONLY for local development
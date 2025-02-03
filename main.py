# Importing necessary libraries and modules
import os  # For interacting with the operating system
import torch  # PyTorch library for deep learning, used here for GPU support
import whisperx  # WhisperX for speech recognition and alignment
import soundfile as sf  # For handling audio files
import logging  # For logging errors and information
from pyannote.audio import Pipeline  # PyAnnote library for speaker diarization
from flask import Flask, render_template, request, redirect, url_for, flash  # Flask modules for building the web app
import numpy as np  # For numerical operations (not used directly here, but imported)
from werkzeug.utils import secure_filename  # For securely handling uploaded file names

# Setting up logging for debugging and monitoring the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initializing the Flask app
app = Flask(__name__)

# Setting the Flask secret key for session management and flash messages
# You should replace the default value with a secure, randomly generated string in production
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "replace_this_with_a_secure_random_string")

# Configuration for the app
HF_TOKEN = os.environ.get("HF_TOKEN", "Your_HF_Token_Goes_Here")  # Hugging Face token for PyAnnote
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac'}  # Supported audio file formats
UPLOAD_FOLDER = 'temp'  # Temporary folder to store uploaded files
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the upload folder if it doesn't exist

# Helper function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to determine the best processing device (CUDA GPU or CPU)
def get_cuda_device():
    if torch.cuda.is_available():  # Check if CUDA-compatible GPU is available
        device_count = torch.cuda.device_count()
        if device_count > 0:
            return "cuda"  # Use GPU if available
    return "cpu"  # Fall back to CPU if no GPU is available

# Main function for transcribing and diarizing an audio file
def transcribe_and_diarize(audio_file_path, device=None):
    # If no device is specified, determine the appropriate device
    if device is None:
        device = get_cuda_device()

    try:
        # Step 1: Load the WhisperX model for transcription
        model = whisperx.load_model("large-v2", device)  # Load a large WhisperX model
        audio = whisperx.load_audio(audio_file_path)  # Load the audio file for processing
        result = model.transcribe(audio, batch_size=16)  # Perform transcription

        # Step 2: Align transcription with timestamps
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)

        # Step 3: Load the PyAnnote speaker diarization pipeline
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",  # Pretrained diarization model
            use_auth_token=HF_TOKEN  # Authentication token for Hugging Face
        ).to(torch.device(device))  # Use the specified device (GPU/CPU)

        # Step 4: Perform speaker diarization on the audio file
        diarization = diarization_pipeline(audio_file_path)

        # Step 5: Combine transcription with speaker information
        diarized_segments = []
        for segment in result["segments"]:
            segment_start = segment["start"]  # Start time of the segment
            segment_end = segment["end"]  # End time of the segment
            
            # Determine the dominant speaker for the segment
            speaker_times = {}  # Dictionary to track speaking duration for each speaker
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if turn.start <= segment_end and turn.end >= segment_start:  # Check for overlap
                    overlap = min(turn.end, segment_end) - max(turn.start, segment_start)
                    if overlap > 0:
                        speaker_times[speaker] = speaker_times.get(speaker, 0) + overlap
            
            # Assign the speaker with the most overlap during the segment
            if speaker_times:
                dominant_speaker = max(speaker_times.items(), key=lambda x: x[1])[0]
            else:
                dominant_speaker = "Unknown Speaker"  # Default label if no speaker is found

            # Format the segment with speaker information
            diarized_segments.append(
                f"[{segment_start:.2f}-{segment_end:.2f}] {dominant_speaker}: {segment['text']}"
            )

        # Return the combined transcription and diarization as a single string
        return "\n".join(diarized_segments)

    except Exception as e:
        # Log and raise any errors encountered during processing
        logger.error(f"Error during processing: {str(e)}")
        raise RuntimeError(f"Error during processing: {str(e)}")

# Flask route for the main page (handles both GET and POST requests)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if an audio file was uploaded
        if "audio_file" not in request.files:
            flash("No file selected!")  # Display an error message to the user
            return redirect(request.url)  # Reload the page

        file = request.files["audio_file"]  # Retrieve the uploaded file
        if file.filename == "":
            flash("No file selected!")  # Check if the file has a name
            return redirect(request.url)

        # Validate the file type
        if not allowed_file(file.filename):
            flash("Invalid file type!")  # Display an error message for unsupported file types
            return redirect(request.url)

        # Save the uploaded file securely to the temporary folder
        filename = secure_filename(file.filename)  # Sanitize the filename
        temp_path = os.path.join(UPLOAD_FOLDER, filename)  # Full path to save the file
        file.save(temp_path)  # Save the file

        try:
            # Transcribe and diarize the uploaded audio file
            transcript = transcribe_and_diarize(temp_path)
        except Exception as e:
            # Handle any errors that occur during processing
            transcript = f"An error occurred: {str(e)}"
        finally:
            # Clean up by removing the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Render the result page with the generated transcript
        return render_template("index.html", transcript=transcript)

    # Render the main page for GET requests
    return render_template("index.html")

# Run the Flask app in debug mode
if __name__ == "__main__":
    app.run(debug=True)

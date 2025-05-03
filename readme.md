# Exact Transcription

A Flask web application that provides accurate audio transcription with speaker diarization using WhisperX and Pyannote.audio. It aims to capture precise timings, pauses, and speaker labels from uploaded audio files.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![Cloud Run](https://img.shields.io/badge/Deploy_with-Cloud_Run-lightgrey.svg)](https://cloud.google.com/run)
[![Cloud Build](https://img.shields.io/badge/Build_with-Cloud_Build-lightgrey.svg)](https://cloud.google.com/build)

## Features

*   Upload common audio formats (MP3, WAV, M4A, etc. - converted internally).
*   Accurate transcription using the Whisper large-v2 model via WhisperX.
*   Word-level timestamp alignment.
*   Speaker diarization (identifying who spoke when) using Pyannote.audio 3.0.
*   Combines transcription and diarization for speaker-labeled output.
*   Simple web interface for uploading files and downloading transcripts.
*   Designed for deployment as a containerized application on Google Cloud Run.

## Tech Stack

*   **Python:** Core programming language.
*   **Flask:** Web framework for the user interface and API.
*   **WhisperX:** Library for fast ASR transcription with word-level timestamps (using faster-whisper).
*   **Pyannote.audio:** Library for speaker diarization.
*   **PyTorch:** Deep learning framework dependency for WhisperX and Pyannote.
*   **Hugging Face Hub:** Used for downloading models and authentication for Pyannote.
*   **Pydub/Librosa/SoundFile:** Audio processing and conversion.
*   **Gunicorn:** Production WSGI server for running Flask.
*   **Docker:** Containerization for packaging the application and dependencies.
*   **Google Cloud Run:** Serverless platform for hosting the container.
*   **Google Cloud Build:** Service for automating container builds and deployments.
*   **Google Secret Manager:** Securely store API keys and secrets.

## Setup and Installation (Local Development)

### Prerequisites

1.  **Python:** Version 3.9 or higher recommended.
2.  **Git:** For cloning the repository.
3.  **FFmpeg:** Required by WhisperX/Pydub for audio processing.
    *   **Windows:** Download from [ffmpeg.org](https://www.ffmpeg.org/download.html) and add the `bin` directory to your system's PATH.
    *   **macOS (using Homebrew):** `brew install ffmpeg`
    *   **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install ffmpeg`
4.  **(Optional) CUDA:** For GPU acceleration (highly recommended for performance). Requires a compatible NVIDIA GPU, CUDA Toolkit, and cuDNN installed. Ensure PyTorch is installed with CUDA support if using GPU.

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pohagan72/Exact_Transcription.git
    cd Exact_Transcription
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv .venv
    .\.venv\Scripts\activate

    # macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration (Local)

1.  **Hugging Face Token:**
    *   You need a Hugging Face account and an Access Token for Pyannote.audio.
    *   Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) to create a token with at least `read` permissions.
    *   **Crucially, you must also accept the user conditions** on the model pages for:
        *   `pyannote/speaker-diarization-3.0` ([link](https://huggingface.co/pyannote/speaker-diarization-3.0))
        *   `pyannote/segmentation-3.0` ([link](https://huggingface.co/pyannote/segmentation-3.0)) - This is a dependency.
        Failure to accept conditions will result in authorization errors even with a valid token.

2.  **Create `.env` file:**
    *   Create a file named `.env` in the project root directory.
    *   Add your secrets to this file:
        ```dotenv
        # .env file
        HUGGING_FACE_HUB_TOKEN=hf_YOUR_HUGGING_FACE_TOKEN
        FLASK_SECRET_KEY=generate_a_strong_random_secret_key
        ```
    *   Replace `hf_YOUR_HUGGING_FACE_TOKEN` with your actual token.
    *   Replace `generate_a_strong_random_secret_key` with a secure random string (used for Flask session management). You can generate one using Python: `python -c 'import secrets; print(secrets.token_hex(16))'`

3.  **Add `.env` to `.gitignore`:**
    *   Ensure your `.gitignore` file contains the line `.env` to prevent accidentally committing your secrets.

## Running Locally

1.  **Ensure your virtual environment is activated.**
2.  **Run the Flask development server:**
    ```bash
    python main.py
    ```
    *(Note: The `main.py` provided runs `app.run()` which uses Flask's development server. For production-like testing, you could use `gunicorn main:app`)*
3.  **Access the application:** Open your web browser and go to `http://127.0.0.1:5000` (or the port specified in the console output).

## Deployment to Google Cloud Run via Cloud Build

This project includes a `Dockerfile` and `cloudbuild.yaml` for automated deployment to Google Cloud Run.

### Prerequisites

1.  **Google Cloud Project:** A GCP project with billing enabled.
2.  **Enabled APIs:** Ensure the following APIs are enabled in your project:
    *   Cloud Build API
    *   Cloud Run API
    *   Secret Manager API
    *   Artifact Registry API (Recommended over Container Registry)
3.  **`gcloud` CLI:** Installed and authenticated ([Google Cloud SDK](https://cloud.google.com/sdk/docs/install)). Run `gcloud auth login` and `gcloud config set project YOUR_PROJECT_ID`.
4.  **Source Code Repository:** Your code pushed to a repository accessible by Cloud Build (e.g., GitHub, Cloud Source Repositories).
5.  **Artifact Registry Repository:** Create a Docker repository in Artifact Registry to store your built images (e.g., `gcloud artifacts repositories create exact-transcription-repo --repository-format=docker --location=us-central1 --description="Docker repository for Exact Transcription app"` - replace location if needed).

### Configuration (Google Cloud)

1.  **Store Secrets in Secret Manager:**
    *   Store your Hugging Face token:
        ```bash
        # Replace YOUR_HF_TOKEN_HERE with your actual token
        echo "hf_YOUR_HF_TOKEN_HERE" | gcloud secrets create huggingface-key --replication-policy="automatic" --data-file=-
        ```
    *   Store your Flask secret key:
        ```bash
        # Replace YOUR_FLASK_SECRET_HERE with your generated secret key
        echo "YOUR_FLASK_SECRET_HERE" | gcloud secrets create flask-secret --replication-policy="automatic" --data-file=-
        ```
    *   **(Important) Grant Access:**
        *   Find your Cloud Build service account email (`PROJECT_NUMBER@cloudbuild.gserviceaccount.com`).
        *   Find/Create the service account Cloud Run will use (default is Compute Engine default SA: `PROJECT_NUMBER-compute@developer.gserviceaccount.com`, but creating a dedicated SA is better practice).
        *   Grant the **Secret Manager Secret Accessor** (`roles/secretmanager.secretAccessor`) role to **both** the Cloud Build service account and the Cloud Run runtime service account on *both* secrets (`huggingface-key` and `flask-secret`). You can do this via the Cloud Console (IAM or Secret Manager UI) or `gcloud secrets add-iam-policy-binding`.

2.  **Configure Cloud Build Service Account Permissions:**
    *   Ensure the Cloud Build service account (`PROJECT_NUMBER@cloudbuild.gserviceaccount.com`) has roles needed for deployment:
        *   `roles/artifactregistry.writer` (to push images)
        *   `roles/run.admin` (to deploy to Cloud Run)
        *   `roles/iam.serviceAccountUser` (to act as the Cloud Run service account during deployment)
        *   `roles/secretmanager.secretAccessor` (if needed during build, but primarily needed by Cloud Run runtime)

### Build & Deploy Steps

1.  **Update `cloudbuild.yaml` (If Needed):**
    *   Ensure the image name paths in `cloudbuild.yaml` match your Project ID and Artifact Registry repository name/location.
    *   Adjust Cloud Run parameters (`--memory`, `--cpu`, `--timeout`, `--region`) as needed. Transcription is resource-intensive; start with generous resources (e.g., `--memory=8Gi`, `--cpu=2`, `--timeout=1200`).

2.  **Trigger the Build:**
    *   **Manual Trigger:** From your project root directory (containing `cloudbuild.yaml`):
        ```bash
        gcloud builds submit --config cloudbuild.yaml .
        ```
    *   **Automated Trigger (Recommended):**
        *   Connect your source repository (e.g., GitHub) to Cloud Build.
        *   Create a Cloud Build trigger that watches your main branch and uses the `cloudbuild.yaml` file. Pushing to the branch will automatically build and deploy.

3.  **Access Deployed App:** Cloud Build will output the URL of your deployed Cloud Run service upon successful completion.

## Environment Variables

| Variable               | Local Source | Cloud Source              | Purpose                                       |
| :--------------------- | :----------- | :------------------------ | :-------------------------------------------- |
| `HUGGING_FACE_HUB_TOKEN` | `.env` file  | Secret Manager (`huggingface-key`) | Authenticates with Hugging Face for Pyannote |
| `FLASK_SECRET_KEY`     | `.env` file  | Secret Manager (`flask-secret`)    | Secures Flask sessions                        |
| `PORT`                 | N/A (uses 5000 default) | Cloud Run (defaults 8080) | Port the Gunicorn server listens on           |

## Known Issues / Limitations

*   **Performance:** Transcription/diarization, especially with `large-v2` on CPU, can be very slow. Using a GPU instance in Cloud Run is recommended for better performance but increases cost.
*   **Resource Usage:** The models require significant RAM. Ensure adequate memory is allocated in Cloud Run (`--memory` flag).
*   **Timeouts:** Long audio files may exceed default request timeouts. Ensure Cloud Run service (`--timeout`) and Gunicorn (`--timeout` in Dockerfile `CMD`) have sufficiently high values (e.g., 1200 seconds or more).
*   **Pyannote Version Warning:** Logs may show warnings about the VAD model being trained on older `pyannote.audio` versions. Functionality seems okay, but results could potentially differ slightly from models trained on newer versions.
*   **Windows Symlinks Warning:** Hugging Face Hub may show warnings about symlinks on Windows during model caching. This does not affect functionality but might increase disk usage in the cache.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## License

(Specify your license here, e.g., MIT License, Apache 2.0, or Unlicensed)
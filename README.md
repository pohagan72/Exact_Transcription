# Exact Transcription
Creates an exact transcription from audio, including pauses and "ah" or "um" type statements
---------------------------------------

Index.html file goes into a subdirectory called Templates under the same location as the python file is placed.

---------------------------------------

Dependencies: pip install torch whisperx soundfile pyannote.audio flask numpy werkzeug huggingface_hub 

Also requires FFmpeg from here if on Windows https://www.ffmpeg.org/download.html 

---------------------------------------

You might need to download cuda dll files directly. If you see an error message that references stranslate2.dll or cudnn_ops_infer64_8.dll then you need to do this.

Instructions found here: https://github.com/SYSTRAN/faster-whisper/discussions/715 

Archive of the cuda files are here: https://developer.nvidia.com/rdp/cudnn-archive#a-collapse811-111. Download the one for cudnn-11.2-windows-x64-v8.1.1.33.zip and then extract the dll's into the folder where ctranslate2.dll is located on your system.

---------------------------------------

The pyannote.audio library may require an older version of huggingface_hub. Specifically, version 0.13.0 or earlier is known to work well with pyannote.audio.
Run the following command to install the compatible version:

pip install huggingface_hub==0.13.0

---------------------------------------

Requires a HuggingFace key for https://huggingface.co/pyannote/speaker-diarization-3.0. You need to accept the terms and create a key that has the "read" options selected or it won't work.

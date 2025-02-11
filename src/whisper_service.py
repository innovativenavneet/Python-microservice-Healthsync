from flask import Flask, request, jsonify
import whisper
import numpy as np
import io
import soundfile as sf
import librosa
import logging
import warnings
import torch
import tempfile

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

app = Flask(__name__)

# Load the model once
model = whisper.load_model("tiny.en", device="cpu")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Whisper AI service is running!"}), 200

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        logger.error("No file provided.")
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    logger.info(f"Received file: {file.filename}, MIME type: {file.mimetype}")

    try:
        file_data = io.BytesIO(file.read())

        # Load using librosa
        audio_data, samplerate = librosa.load(file_data, sr=16000, mono=True)
        logger.info(f"Loaded audio: {len(audio_data)} samples at {samplerate} Hz")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, audio_data, samplerate)
            logger.info("Temp WAV file created.")

            # Transcribe using Whisper
            result = model.transcribe(tmpfile.name, fp16=False)
            logger.info(f"Transcription result: {result['text']}")

        return jsonify({"text": result['text']})

    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": "An error occurred during transcription.", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

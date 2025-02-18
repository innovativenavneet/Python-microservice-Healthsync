from flask import Flask, request, jsonify
import numpy as np
import io
import soundfile as sf
import librosa
import logging
import warnings
import torch
import os
from model_loader import model  # Import the model from modelloader.py

# Suppress specific warning
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

# Ensure CUDA is disabled
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Whisper AI service is running!"}), 200

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        logger.error("No file provided in the request.")
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    logger.info(f"Received file: {file.filename} with MIME type: {file.mimetype}")

    try:
        audio_data, samplerate = sf.read(io.BytesIO(file.read()), dtype="float32")
        logger.info(f"Audio loaded. Sample rate: {samplerate}, Length: {len(audio_data)} samples.")

        # Resample to 16kHz if needed
        if samplerate != 16000:
            logger.info(f"Resampling audio from {samplerate} Hz to 16000 Hz...")
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)

        # Normalize audio
        audio_data = librosa.util.normalize(audio_data)
        logger.info("Audio normalization complete.")

        # Ensure correct NumPy format
        audio_data = np.array(audio_data, dtype=np.float32)

        # Transcribe
        logger.info("Starting transcription...")
        result = model.transcribe(audio_data)
        logger.info(f"Transcription completed: {result['text']}")

        return jsonify({"text": result['text']})

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return jsonify({"error": "An error occurred.", "details": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)

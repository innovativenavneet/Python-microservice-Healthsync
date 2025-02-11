from flask import Flask, request, jsonify
import whisper
import numpy as np
import io
import soundfile as sf
import librosa
import logging
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

app = Flask(__name__)
model = whisper.load_model("base")

# Setup logging for debugging purposes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Whisper AI service is running!"}), 200


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    # Check if file is present in the request
    if 'file' not in request.files:
        logger.error("No file provided in the request.")
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    logger.info(f"Received file: {file.filename} with MIME type: {file.mimetype}")

    try:
        # Read the audio file into memory
        audio_data, samplerate = sf.read(io.BytesIO(file.read()), dtype="float32")  # Ensure dtype is float32
        logger.info(f"Audio file loaded. Samplerate: {samplerate}, Length of audio data: {len(audio_data)} samples.")

        # Resample audio if necessary to 16kHz for Whisper
        if samplerate != 16000:
            logger.info(f"Resampling audio from {samplerate} Hz to 16000 Hz...")
            audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=16000)
            samplerate = 16000

        # Normalize audio data to ensure consistent levels
        audio_data = librosa.util.normalize(audio_data)
        logger.info("Audio normalization complete.")

        # Ensure audio data is in the correct dtype for Whisper (float32)
        audio_data = audio_data.astype(np.float32)

        # Perform transcription using Whisper model
        logger.info("Starting transcription process with Whisper model...")
        result = model.transcribe(audio_data)
        logger.info(f"Transcription completed: {result['text']}")

        # Return the transcription result
        return jsonify({"text": result['text']})

    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return jsonify({"error": "An error occurred during transcription.", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
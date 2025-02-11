from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load the Whisper model
model = whisper.load_model("base")
# Suppress the FutureWarning
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    result = model.transcribe(audio_file)

    return jsonify({"text": result['text']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # Run on port 5001

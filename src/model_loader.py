import os
import whisper

MODEL_PATH = "/tmp/whisper-model"
MODEL_NAME = "base.en"

os.makedirs(MODEL_PATH, exist_ok=True)

model_assets_path = os.path.join(MODEL_PATH, "whisper-assets")
if not os.path.exists(model_assets_path):
    print(f"Downloading Whisper model ({MODEL_NAME})...")
    model = whisper.load_model(MODEL_NAME, download_root=MODEL_PATH, device="cpu")
    print("Model downloaded and cached.")
else:
    print("Loading cached model...")
    model = whisper.load_model(MODEL_NAME, download_root=MODEL_PATH, device="cpu")

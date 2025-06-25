#app.py
from flask import Flask, request, jsonify
from pp import extract_all_features, predict_audio
import os

app = Flask(__name__)
@app.route("/", methods=["GET"])
def index():
    return "Lung Disease Classifier API is running", 200

@app.route("/extract", methods=["POST"])
def extract():
    if 'audio' not in request.files:
        return jsonify({"status": "error", "message": "No audio file provided"}), 400

    file = request.files['audio']
    audio_path = os.path.join("uploads", "input.wav")
    os.makedirs("uploads", exist_ok=True)
    file.save(audio_path)

    # Feature Extraction
    feature_result = extract_all_features(audio_path)
    if feature_result["status"] != "success":
        return jsonify(feature_result), 500

    features = feature_result["features"]  # Already a list of float values
    prediction_result = predict_audio(features)

    return jsonify(prediction_result)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"✅ Flask app running on port {port}")
    app.run(host="0.0.0.0", port=port)


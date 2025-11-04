import os
import logging
from flask import Flask, request, jsonify
from google.cloud import aiplatform
import base64
import tempfile
from pydub import AudioSegment

# ---------- CONFIG ----------
VERTEX_ENDPOINT_ID = os.getenv("VERTEX_ENDPOINT_ID")  # e.g. projects/123/locations/us-central1/endpoints/456
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION", "us-central1")

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Vertex AI client ----------
aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(endpoint_name=VERTEX_ENDPOINT_ID)


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ready"}), 200


@app.route("/speech2text", methods=["POST"])
def speech2text():
    try:
        data = request.get_json(silent=True) or {}
        audio_b64 = data.get("audio_base64")
        fmt = data.get("format", "wav")
        if not audio_b64:
            return jsonify({"error": "missing 'audio_base64'"}), 400

        # Convert to WAV if needed
        audio_bytes = base64.b64decode(audio_b64)
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            audio.export(tmp.name, format="wav")
            with open(tmp.name, "rb") as f:
                content = base64.b64encode(f.read()).decode("utf-8")

        # Call Vertex endpoint
        instance = {"audio_base64": content, "format": "wav"}
        response = endpoint.predict(instances=[instance])
        logger.info("Vertex response: %s", response)

        return jsonify({"text": response.predictions[0].get("text", "")})
    except Exception as e:
        logger.exception("ASR processing failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

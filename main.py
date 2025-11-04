import os
import io
import logging
import base64
import tempfile
from flask import Flask, request, jsonify
from google.cloud import aiplatform
from pydub import AudioSegment

# ---------- CONFIG ----------
PROJECT_ID = os.getenv("PROJECT_ID", "burmese-voice")
REGION = os.getenv("REGION", "us-central1")
VERTEX_ENDPOINT_ID = os.getenv(
    "VERTEX_ENDPOINT_ID",
    "projects/burmese-voice/locations/us-central1/endpoints/7279126516179402752"
)

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
        content_type = request.headers.get("Content-Type", "")

        # ✅ Case 1: multipart/form-data (Postman or Chatrace)
        if "multipart/form-data" in content_type:
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "Missing uploaded file"}), 400
            audio_bytes = file.read()
            fmt = file.filename.split(".")[-1]
            logger.info("Received file via multipart: %s (%d bytes)", file.filename, len(audio_bytes))

        # ✅ Case 2: raw binary body (audio/wav, audio/m4a)
        elif content_type.startswith("audio/"):
            audio_bytes = request.data
            fmt = content_type.split("/")[-1]
            logger.info("Received raw audio stream (%s, %d bytes)", fmt, len(audio_bytes))

        # ✅ Case 3: JSON with base64
        else:
            data = request.get_json(silent=True) or {}
            audio_b64 = data.get("audio_base64")
            fmt = data.get("format", "wav")
            if not audio_b64:
                return jsonify({"error": "missing 'audio_base64'"}), 400
            audio_bytes = base64.b64decode(audio_b64)

        # ✅ Convert to WAV (Vertex expects normalized WAV)
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            audio.export(tmp.name, format="wav")
            with open(tmp.name, "rb") as f:
                content = base64.b64encode(f.read()).decode("utf-8")

        # ✅ Call Vertex endpoint
        instance = {"audio_base64": content, "src_lang": "mya", "tgt_lang": "mya"}
        response = endpoint.predict(instances=[instance])
        logger.info("Vertex response: %s", response)

        # ✅ Return formatted response
        return jsonify(response.predictions)

    except Exception as e:
        logger.exception("ASR processing failed")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

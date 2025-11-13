import os
import io
import logging
import base64
import tempfile
import requests
from flask import Flask, request, jsonify
from google.cloud import aiplatform
from pydub import AudioSegment, utils

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_ID = os.getenv("PROJECT_ID", "burmese-voice")
REGION = os.getenv("REGION", "us-central1")
VERTEX_ENDPOINT_ID = os.getenv(
    "VERTEX_ENDPOINT_ID",
    "projects/burmese-voice/locations/us-central1/endpoints/7279126516179402752"
)

# -----------------------------
# Flask + Logging Setup
# -----------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asr")

# -----------------------------
# Ensure ffmpeg/ffprobe paths
# -----------------------------
AudioSegment.converter = utils.which("ffmpeg") or "/usr/bin/ffmpeg"
AudioSegment.ffprobe = utils.which("ffprobe") or "/usr/bin/ffprobe"

# -----------------------------
# Vertex AI Client
# -----------------------------
aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(endpoint_name=VERTEX_ENDPOINT_ID)


# -----------------------------
# Health Check
# -----------------------------
@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ready"}), 200


# -----------------------------
# Speech → Text API
# -----------------------------
@app.route("/speech2text", methods=["POST"])
def speech2text():
    try:
        content_type = request.headers.get("Content-Type", "")
        fmt = "wav"
        audio_bytes = None

        # Case 1: multipart/form-data
        if "multipart/form-data" in content_type:
            file = request.files.get("file")
            if not file:
                return jsonify({"error": "Missing uploaded file"}), 400
            fmt = file.filename.split(".")[-1].lower()
            audio_bytes = file.read()
            logger.info("📁 Multipart file: %s (%d bytes)", file.filename, len(audio_bytes))

        # Case 2: raw binary
        elif content_type.startswith("audio/"):
            fmt = content_type.split("/")[-1]
            audio_bytes = request.data
            logger.info("📡 Raw audio stream (%s, %d bytes)", fmt, len(audio_bytes))

        # Case 3: JSON base64 or URL
        else:
            data = request.get_json(silent=True) or {}
            fmt = data.get("format", "m4a")
            audio_b64 = data.get("audio_base64")
            audio_url = data.get("url")

            if audio_url:
                logger.info(f"🌐 Downloading from URL: {audio_url}")
                resp = requests.get(audio_url)
                if resp.status_code != 200:
                    return jsonify({"error": f"Failed to download audio from URL ({resp.status_code})"}), 400
                audio_bytes = resp.content
            elif audio_b64:
                audio_bytes = base64.b64decode(audio_b64)
            else:
                return jsonify({"error": "Missing 'audio_base64' or 'url'"}), 400

        # Convert input to WAV at 16 kHz mono
        with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            tmp_out_path = tmp_out.name

        try:
            audio = AudioSegment.from_file(tmp_in_path, format=fmt)
            # 🔥 Force 16 kHz mono conversion
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(tmp_out_path, format="wav")
            logger.info("🎵 Converted %s → 16 kHz mono WAV", fmt)
        except Exception as e:
            logger.warning("⚠️ Conversion fallback: %s", e)
            tmp_out_path = tmp_in_path

        # Encode to base64
        with open(tmp_out_path, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf-8")

        # Clean up
        for path in [tmp_in_path, tmp_out_path]:
            if os.path.exists(path):
                os.remove(path)

        # Send to Vertex endpoint
        instance = {"audio_base64": content, "src_lang": "mya", "tgt_lang": "mya"}
        response = endpoint.predict(instances=[instance])
        logger.info("✅ Vertex response received")

        return jsonify({
            "status": "success",
            "predictions": response.predictions
        })

    except Exception as e:
        logger.exception("❌ ASR processing failed")
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Root
# -----------------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "✅ Burmese ASR API on Cloud Run", "endpoint": VERTEX_ENDPOINT_ID})


# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

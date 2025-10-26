import os
import logging
import threading
import time
from threading import Lock
from flask import Flask, request, jsonify
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
import base64
import io
import soundfile as sf
from pydub import AudioSegment

# ---------- CONFIG ----------
WORKSPACE = os.getenv("TRANSFORMERS_CACHE", "/workspace/hf_cache")
os.environ["TRANSFORMERS_CACHE"] = WORKSPACE
os.makedirs(WORKSPACE, exist_ok=True)

ASR_BUCKET = "speechtotext-model-bucket"
ASR_PREFIX = "model/finetuned-seamlessm4t-burmese"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
asr_lock = Lock()

# Global model objects + flags
asr_pipe = None
model_loading = False
model_ready = False
model_load_error = None

# ---------- Helper: parallel GCS download ----------
def download_from_gcs(bucket_name, prefix, local_dir, max_workers=8):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        logger.warning("⚠️ No files found in gs://%s/%s", bucket_name, prefix)
        return

    os.makedirs(local_dir, exist_ok=True)

    def _dl(blob):
        if blob.size == 0:
            return
        rel = os.path.relpath(blob.name, prefix)
        if rel.startswith("checkpoint/"):
            return
        dest = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)
        logger.info("📦 %s → %s", blob.name, dest)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_dl, b) for b in blobs]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.error("Failed to download a blob: %s", e)

    logger.info("✅ Finished downloading %d files from gs://%s/%s", len(blobs), bucket_name, prefix)

# ---------- Load ASR Model ----------
def load_asr_model(local_path):
    global asr_pipe
    logger.info("Loading ASR model from %s", local_path)

    processor = AutoProcessor.from_pretrained(local_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        local_path,
        torch_dtype=torch.float32,
        device_map=None,
        low_cpu_mem_usage=True
    )

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor
    )
    logger.info("✅ ASR model loaded successfully")

# ---------- Background preload ----------
def preload_asr_background():
    global model_loading, model_ready, model_load_error
    with asr_lock:
        if model_ready or model_loading:
            return
        model_loading = True

    try:
        start = time.time()
        local_asr_dir = os.path.join(WORKSPACE, "asr")

        # Download model if missing
        if not os.path.exists(local_asr_dir) or not os.listdir(local_asr_dir):
            logger.info("⬇️ Downloading ASR model from GCS to %s", local_asr_dir)
            download_from_gcs(ASR_BUCKET, ASR_PREFIX, local_asr_dir, max_workers=8)
        else:
            logger.info("ASR model already present at %s", local_asr_dir)

        # Load into memory
        load_asr_model(local_asr_dir)
        model_ready = True
        logger.info("✅ ASR model ready (preload took %.1f s)", time.time() - start)

    except Exception as e:
        model_load_error = str(e)
        logger.exception("❌ Failed to preload ASR model: %s", e)
    finally:
        model_loading = False

def start_preload_thread():
    t = threading.Thread(target=preload_asr_background, daemon=True)
    t.start()
    return t

# Start preload right away (cold start)
start_preload_thread()

# ---------- Endpoints ----------
@app.route("/healthz", methods=["GET"])
def healthz():
    status = "ready" if model_ready else ("loading" if model_loading else "not_loaded")
    return jsonify({"status": status}), 200 if model_ready else 503

@app.route("/speech2text", methods=["POST"])
def speech2text():
    if not model_ready:
        return jsonify({
            "error": "model loading",
            "loading": model_loading,
            "error_details": model_load_error
        }), 503

    data = request.get_json(silent=True) or {}
    audio_b64 = data.get("audio_base64")
    fmt = data.get("format", "wav")

    if not audio_b64:
        return jsonify({"error": "missing 'audio_base64'"}), 400

    try:
        audio_bytes = base64.b64decode(audio_b64)
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            audio.export(tmp.name, format="wav")
            text = asr_pipe(tmp.name)["text"]

        return jsonify({"text": text})
    except Exception as e:
        logger.exception("ASR processing failed")
        return jsonify({"error": str(e)}), 500

# ---------- Entrypoint ----------
if __name__ == "__main__":
    if torch.cuda.is_available():
        logger.info("🔥 GPU available: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("⚠️ GPU not available — using CPU")

    logger.info("Starting ASR Flask app on 0.0.0.0:8080 (model preloading in background)")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

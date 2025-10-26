import os
import logging
import threading
import time
from threading import Lock
from flask import Flask, request, jsonify
import tempfile
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# audio libs used only by speech endpoints (keep if you need them)
import requests
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
QA_BUCKET = "qa-model-bucket"
QA_PREFIX = "model/finetuned-qa-burmese"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
asr_lock = Lock()
qa_lock = Lock()

# global model objects + state flags
qa_pipe = None
qa_model = None
qa_tokenizer = None
model_loading = False
model_ready = False
model_load_error = None

# ---------- download helper (parallel) ----------
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

# ---------- QA loader (memory-friendly) ----------
def load_qa_model(local_path):
    global qa_pipe, qa_model, qa_tokenizer
    # Use memory-saving options. device_map="auto" lets accelerate/offload if available.
    logger.info("Loading QA tokenizer/model from %s", local_path)
    qa_tokenizer = AutoTokenizer.from_pretrained(local_path, use_fast=False)
    qa_model = AutoModelForSeq2SeqLM.from_pretrained(
        local_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    qa_pipe = pipeline("text2text-generation", model=qa_model, tokenizer=qa_tokenizer)
    logger.info("✅ QA model loaded into memory")

# ---------- Background preload ----------
def preload_models_background():
    global model_loading, model_ready, model_load_error
    with qa_lock:
        if model_ready or model_loading:
            return
        model_loading = True

    try:
        start = time.time()
        # download QA model files to cache
        local_qa_dir = os.path.join(WORKSPACE, "qa")
        if not os.path.exists(local_qa_dir) or not os.listdir(local_qa_dir):
            logger.info("⬇️ Downloading QA model from GCS to %s", local_qa_dir)
            download_from_gcs(QA_BUCKET, QA_PREFIX, local_qa_dir, max_workers=8)
        else:
            logger.info("QA files already present at %s", local_qa_dir)

        # load model (memory-friendly)
        load_qa_model(local_qa_dir)
        model_ready = True
        logger.info("Model ready (preload took %.1f s)", time.time() - start)
    except Exception as e:
        model_load_error = str(e)
        logger.exception("Failed to preload model: %s", e)
    finally:
        model_loading = False

# Kick off preload in a daemon thread at process start
def start_preload_thread():
    t = threading.Thread(target=preload_models_background, daemon=True)
    t.start()
    return t

# Start preload right away (cold start)
start_preload_thread()

# ---------- Endpoints ----------
@app.route("/healthz", methods=["GET"])
def healthz():
    status = "ready" if model_ready else ("loading" if model_loading else "not_loaded")
    return jsonify({"status": status}), 200 if model_ready else 503

@app.route("/textqa", methods=["POST"])
def textqa():
    if not model_ready:
        # short-circuit: model still loading
        return jsonify({"error": "model loading", "loading": model_loading, "error_details": model_load_error}), 503

    data = request.get_json(silent=True) or {}
    question = data.get("text") or data.get("question") or ""
    if not question:
        return jsonify({"error": "missing 'text' or 'question'"}), 400

    try:
        # run qa via the pipeline
        result = qa_pipe(f"question: {question}\nAnswer:")
        # pipeline returns list of dicts; check shape
        answer = result[0].get("generated_text") if isinstance(result, list) else result.get("generated_text")
        return jsonify({"answer": answer})
    except Exception as e:
        logger.exception("Error running QA")
        return jsonify({"error": str(e)}), 500

# other endpoints (asr/speechqa) should also check model_ready similarly before using QA model

if __name__ == "__main__":
    # log device & memory
    if torch.cuda.is_available():
        try:
            logger.info("🔥 GPU available: %s", torch.cuda.get_device_name(0))
        except Exception:
            logger.info("🔥 GPU available")
    else:
        logger.warning("⚠️ GPU not available — using CPU")

    logger.info("Starting Flask (dev) on 0.0.0.0:8080 — preloading models in background")
    # NOTE: in production run with gunicorn as shown below in Dockerfile recommendations
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

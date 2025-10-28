# ===== BASE IMAGE: CUDA runtime =====
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV TRANSFORMERS_CACHE=/workspace/hf_cache
RUN mkdir -p /workspace/hf_cache

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3.11-distutils python3-pip \
    git curl ffmpeg && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

COPY requirements.txt .

# ✅ Install latest PyTorch compatible with transformers >= 4.40
RUN pip install --no-cache-dir torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

COPY . .

# ===== PRELOAD MODEL INTO IMAGE =====
COPY model/finetuned-seamlessm4t-burmese /workspace/hf_cache/asr


EXPOSE 8080
# single process, threaded worker — adjust threads/workers to your CPU & memory
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:8080", "main:app", "--timeout", "300"]

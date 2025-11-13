# Start from a lightweight official Python image
FROM python:3.11-slim

# Prevent Python from buffering stdout/stderr
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set working directory
WORKDIR /app

# ---- Install System Dependencies ----
# ffmpeg provides both ffmpeg and ffprobe for audio conversion
# Clean up after install to keep the image small
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python Dependencies ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy Application Code ----
COPY . .

# ---- Expose and Start ----
EXPOSE 8080
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "4", "-b", "0.0.0.0:8080", "main:app", "--timeout", "300"]


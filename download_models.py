import os
from google.cloud import storage

def download_from_gcs(bucket_name, prefix, local_dir):
    print(f"📦 Downloading {prefix} from {bucket_name} → {local_dir}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))
    if not blobs:
        print(f"⚠️ No files found in gs://{bucket_name}/{prefix}")
        return
    os.makedirs(local_dir, exist_ok=True)
    for blob in blobs:
        if blob.size == 0:
            continue
        rel_path = os.path.relpath(blob.name, prefix)
        dest_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob.download_to_filename(dest_path)
        print("✅", blob.name)
    print(f"🎯 Finished downloading {len(blobs)} files from {bucket_name}/{prefix}\n")

#download_from_gcs("qa-model-bucket", "model/finetuned-qa-burmese", "/workspace/qa")
download_from_gcs("speechtotext-model-bucket", "model/finetuned-seamlessm4t-burmese", "/workspace/asr")

# upload_demos.py
#
# Run once (or whenever you add new sample images) to push everything in
# your local demos/ folder up to Firebase Storage under demos/.
#
# Usage: python upload_demos.py
#
# Reads the same credentials as the Streamlit app, but from a local
# .streamlit/secrets.toml (this script is meant to be run from your machine,
# not from Streamlit Cloud).

import os
import sys

try:
    import toml
except ImportError:
    print("Installing missing dependency 'toml'...")
    os.system(f"{sys.executable} -m pip install toml")
    import toml

import firebase_admin
from firebase_admin import credentials, storage

SECRETS_PATH = os.path.join(".streamlit", "secrets.toml")
LOCAL_DEMOS_DIR = "demos"
STORAGE_PREFIX = "demos/"


def main():
    if not os.path.exists(SECRETS_PATH):
        print(f"Could not find {SECRETS_PATH}. Copy secrets.toml.example there and fill it in first.")
        sys.exit(1)

    secrets = toml.load(SECRETS_PATH)
    if "firebase" not in secrets or "firebase_bucket" not in secrets:
        print("secrets.toml is missing the [firebase] section or firebase_bucket key.")
        sys.exit(1)

    cred = credentials.Certificate(secrets["firebase"])
    firebase_admin.initialize_app(cred, {"storageBucket": secrets["firebase_bucket"]})
    bucket = storage.bucket()

    if not os.path.isdir(LOCAL_DEMOS_DIR):
        print(f"No local '{LOCAL_DEMOS_DIR}' directory found.")
        sys.exit(1)

    uploaded = 0
    for filename in sorted(os.listdir(LOCAL_DEMOS_DIR)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        local_path = os.path.join(LOCAL_DEMOS_DIR, filename)
        remote_path = STORAGE_PREFIX + filename
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_path)
        print(f"Uploaded {local_path} -> gs://{bucket.name}/{remote_path}")
        uploaded += 1

    print(f"\nDone. Uploaded {uploaded} image(s) to gs://{bucket.name}/{STORAGE_PREFIX}")


if __name__ == "__main__":
    main()
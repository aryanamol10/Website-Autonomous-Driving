# firebase_client.py
#
# Central place that talks to Firebase. Credentials come from Streamlit's
# secrets store (st.secrets), so nothing sensitive lives in the repo.
#
# Local dev:      put your service account fields in .streamlit/secrets.toml
# Streamlit Cloud: paste the same TOML into the app's "Secrets" settings panel
#
# Storage layout expected in the bucket:
#   demos/<label>.jpg   (or .png) -- one object per demo dataset image
#
# Firestore layout used for logging predictions:
#   predictions/{auto-id}: { class, confidence, severity, source, timestamp }

import io
import json

import streamlit as st
import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image


@st.cache_resource
def init_firebase():
    """Initializes the Firebase Admin app exactly once per process."""
    if firebase_admin._apps:
        return firebase_admin.get_app()

    if "firebase" not in st.secrets:
        st.sidebar.warning(
            "No Firebase credentials found in st.secrets['firebase']. "
            "Dataset gallery and prediction logging will be unavailable."
        )
        return None

    firebase_secrets = dict(st.secrets["firebase"])
    cred = credentials.Certificate(firebase_secrets)

    bucket_name = st.secrets.get("firebase_bucket", None)
    return firebase_admin.initialize_app(
        cred,
        {"storageBucket": bucket_name} if bucket_name else None,
    )


def get_firestore_client():
    app = init_firebase()
    if app is None:
        return None
    return firestore.client()


def get_bucket():
    app = init_firebase()
    if app is None:
        return None
    try:
        return storage.bucket()
    except ValueError:
        st.sidebar.warning(
            "Firebase Storage bucket not configured. Set 'firebase_bucket' "
            "in st.secrets to your bucket name (e.g. 'your-project.appspot.com')."
        )
        return None


@st.cache_data(ttl=300, show_spinner=False)
def list_demo_datasets(prefix: str = "demos/"):
    """Returns a list of (label, storage_path) for every image under `prefix`."""
    bucket = get_bucket()
    if bucket is None:
        return []

    datasets = []
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name == prefix or blob.name.endswith("/"):
            continue
        filename = blob.name.rsplit("/", 1)[-1]
        label = filename.rsplit(".", 1)[0].replace("_", " ").title()
        datasets.append((label, blob.name))
    return datasets


@st.cache_data(ttl=300, show_spinner=False)
def download_image(storage_path: str):
    """Downloads a Storage object and returns it as a PIL Image, or None."""
    bucket = get_bucket()
    if bucket is None:
        return None
    blob = bucket.blob(storage_path)
    if not blob.exists():
        return None
    data = blob.download_as_bytes()
    return Image.open(io.BytesIO(data)).convert("RGB")

@st.cache_data(ttl=300, show_spinner=False)
def list_datasets(prefix: str = "datasets/"):
    """Returns the distinct dataset names found under `prefix`.

    Expects layout: datasets/<dataset_name>/<split>/<class>/<file>.jpg
    """
    bucket = get_bucket()
    if bucket is None:
        return []

    names = set()
    for blob in bucket.list_blobs(prefix=prefix):
        rel = blob.name[len(prefix):]
        if not rel or "/" not in rel:
            continue
        dataset_name = rel.split("/", 1)[0]
        names.add(dataset_name)
    return sorted(names)


@st.cache_data(ttl=300, show_spinner=False)
def list_dataset_images(dataset_name: str, split: str = "test", prefix: str = "datasets/"):
    """Returns (label, storage_path) for every image under
    datasets/<dataset_name>/<split>/, where label is the class subfolder.
    """
    bucket = get_bucket()
    if bucket is None:
        return []

    folder_prefix = f"{prefix}{dataset_name}/{split}/"
    images = []
    for blob in bucket.list_blobs(prefix=folder_prefix):
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(folder_prefix):]
        parts = rel.split("/")
        label = parts[0] if len(parts) > 1 else "unlabeled"
        images.append((label, blob.name))
    return images
    
def log_prediction(prediction_text, confidence, severity_label, source):
    """Writes one prediction record to Firestore. Safe no-op if unconfigured."""
    db = get_firestore_client()
    if db is None:
        return
    try:
        db.collection("predictions").add(
            {
                "class": prediction_text,
                "confidence": float(confidence),
                "severity": severity_label,
                "source": source,
                "timestamp": firestore.SERVER_TIMESTAMP,
            }
        )
    except Exception as e:
        st.sidebar.warning(f"Could not log prediction to Firestore: {e}")
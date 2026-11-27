"""
Download Dlib's 68-point facial landmark model into models/.

Usage:
    python scripts/download_model.py

The model (about 100 MB) is not stored in Git: it is close to GitHub's 100 MB
per-file limit, it is freely downloadable, and its training data (iBUG 300-W)
does not allow commercial use. This script fetches the compressed file
(about 64 MB) from dlib.net and decompresses it. Uses only the standard library.
"""

import bz2
import os
import sys
import urllib.request

URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(REPO_DIR, "models", "shape_predictor_68_face_landmarks.dat")
EXPECTED_SIZE = 99_693_937  # bytes, size of the official decompressed file


def show_progress(blocks, block_size, total):
    if total > 0:
        done = min(blocks * block_size, total)
        print(f"\r  downloading... {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)


def main():
    if os.path.isfile(MODEL_PATH) and os.path.getsize(MODEL_PATH) == EXPECTED_SIZE:
        print(f"[OK] Model already present: {MODEL_PATH}")
        return

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    archive = MODEL_PATH + ".bz2"
    try:
        urllib.request.urlretrieve(URL, archive, show_progress)
        print()
    except OSError as exc:
        sys.exit(f"\n[ERROR] Download failed ({exc}). Check your internet connection, "
                 f"or download {URL} manually (see README).")

    print("  decompressing...")
    # Stream in chunks so the whole file never has to sit in memory.
    with bz2.open(archive, "rb") as src, open(MODEL_PATH, "wb") as dst:
        while chunk := src.read(1024 * 1024):
            dst.write(chunk)
    os.remove(archive)

    size = os.path.getsize(MODEL_PATH)
    if size != EXPECTED_SIZE:
        print(f"[WARNING] Unexpected file size ({size} bytes); the download may be corrupt.")
    print(f"[OK] Saved {MODEL_PATH}")


if __name__ == "__main__":
    main()

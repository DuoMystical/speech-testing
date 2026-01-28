#!/usr/bin/env python3
"""
Download TEN VAD ONNX model (Silero VAD).

TEN VAD uses Silero VAD internally - a lightweight, fast, and accurate
voice activity detector optimized for real-time streaming applications.

Usage:
    python download_ten_vad.py --output ./models/ten_vad
"""

import argparse
from pathlib import Path
import urllib.request
import shutil
import ssl


def download_ten_vad(output_dir: str):
    """Download Silero VAD ONNX model (used by TEN VAD)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Silero VAD v5 ONNX model - official source from snakers4/silero-vad
    # This is the same model used by TEN framework internally
    urls_to_try = [
        # Primary: Official Silero VAD repository
        "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx",
        # Fallback: HuggingFace mirror
        "https://huggingface.co/onnx-community/silero-vad/resolve/main/silero_vad.onnx",
    ]

    onnx_path = output_path / "ten_vad.onnx"

    # Create SSL context that doesn't verify (for environments with cert issues)
    ssl_context = ssl.create_default_context()

    downloaded = False
    for url in urls_to_try:
        print(f"Attempting download from: {url}")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ssl_context, timeout=60) as response:
                with open(onnx_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            print(f"Downloaded successfully: {onnx_path}")
            downloaded = True
            break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            continue

    if not downloaded:
        raise RuntimeError("Failed to download Silero VAD from all sources")

    # Create config file
    config_path = output_path / "config.json"
    import json
    config = {
        "model_name": "Silero-VAD",
        "model_version": "v5",
        "sample_rate": 16000,
        "frame_size_samples": 512,  # 32ms at 16kHz (Silero VAD requirement)
        "hop_size_samples": 512,
        "threshold": 0.5,
        "min_speech_duration_ms": 64,
        "min_silence_duration_ms": 320,
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nDownload complete! Files saved to: {output_path}")
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TEN VAD ONNX model")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./models/ten_vad",
        help="Output directory"
    )

    args = parser.parse_args()
    download_ten_vad(args.output)

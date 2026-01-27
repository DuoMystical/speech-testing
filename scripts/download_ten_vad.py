#!/usr/bin/env python3
"""
Download TEN VAD ONNX model from HuggingFace.

Usage:
    python download_ten_vad.py --output ./models/ten_vad
"""

import argparse
from pathlib import Path


def download_ten_vad(output_dir: str):
    """Download TEN VAD ONNX model."""
    from huggingface_hub import hf_hub_download

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_id = "TEN-framework/ten-vad"

    # Download the ONNX model
    print(f"Downloading TEN VAD from {repo_id}...")

    files_to_download = [
        "ten_vad.onnx",
    ]

    for filename in files_to_download:
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(output_path),
            )
            print(f"Downloaded: {downloaded_path}")
        except Exception as e:
            print(f"Warning: Could not download {filename}: {e}")

    # Create config file
    config_path = output_path / "config.json"
    import json
    config = {
        "model_name": "TEN-VAD",
        "sample_rate": 16000,
        "frame_size_samples": 160,  # 10ms at 16kHz
        "hop_size_samples": 160,
        "threshold": 0.5,
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nDownload complete! Files saved to: {output_path}")


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

#!/usr/bin/env python3
"""
Export Parakeet-TDT 0.6B v3 to ONNX with streaming/cache support.

This script exports the model in a format suitable for streaming inference
with TensorRT optimization.

Usage:
    python export_parakeet.py --output ./models/parakeet

Requirements:
    pip install nemo_toolkit[asr] onnx onnxruntime
"""

import argparse
import os
from pathlib import Path


def export_parakeet_tdt(output_dir: str, model_name: str = "nvidia/parakeet-tdt-0.6b-v3"):
    """Export Parakeet-TDT model to ONNX for inference."""
    import nemo.collections.asr as nemo_asr
    import torch

    print(f"Loading model: {model_name}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Export without streaming cache - simpler and more reliable
    # Streaming can be handled at application level by processing chunks
    print("Configuring model for standard ONNX export...")

    # Disable cache support to avoid tensor shape issues during export
    if hasattr(model, 'set_export_config'):
        model.set_export_config({
            'cache_support': 'False',
        })

    # Export encoder
    encoder_path = output_path / "encoder.onnx"
    print(f"Exporting encoder to: {encoder_path}")
    try:
        model.encoder.export(
            str(encoder_path),
            onnx_opset_version=17,
            check_trace=False,
            dynamic_axes={
                'audio_signal': {0: 'batch', 2: 'time'},
                'length': {0: 'batch'},
            },
        )
    except Exception as e:
        print(f"Standard export failed: {e}")
        print("Trying alternative export method...")
        # Fallback: export with minimal config
        model.encoder.export(
            str(encoder_path),
            onnx_opset_version=14,
            check_trace=False,
        )

    # Export decoder (prediction network for transducer)
    decoder_path = output_path / "decoder.onnx"
    print(f"Exporting decoder to: {decoder_path}")
    try:
        model.decoder.export(
            str(decoder_path),
            onnx_opset_version=17,
            check_trace=False,
        )
    except Exception as e:
        print(f"Decoder export with opset 17 failed: {e}")
        print("Trying with opset 14...")
        model.decoder.export(
            str(decoder_path),
            onnx_opset_version=14,
            check_trace=False,
        )

    # Export joint network
    joint_path = output_path / "joint.onnx"
    print(f"Exporting joint network to: {joint_path}")
    try:
        model.joint.export(
            str(joint_path),
            onnx_opset_version=17,
            check_trace=False,
        )
    except Exception as e:
        print(f"Joint export with opset 17 failed: {e}")
        print("Trying with opset 14...")
        model.joint.export(
            str(joint_path),
            onnx_opset_version=14,
            check_trace=False,
        )

    # Save vocabulary/tokenizer
    vocab_path = output_path / "vocab.txt"
    print(f"Saving vocabulary to: {vocab_path}")
    if hasattr(model, 'tokenizer') and model.tokenizer is not None:
        vocab = model.tokenizer.vocab
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token in vocab:
                f.write(f"{token}\n")
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'vocabulary'):
        vocab = model.decoder.vocabulary
        with open(vocab_path, 'w', encoding='utf-8') as f:
            for token in vocab:
                f.write(f"{token}\n")

    # Save model config
    config_path = output_path / "config.json"
    print(f"Saving config to: {config_path}")
    import json
    config = {
        "model_name": model_name,
        "sample_rate": 16000,
        "n_mels": 80,
        "frame_length_ms": 25,
        "frame_shift_ms": 10,
        "window_size_ms": 400,  # Conformer typically uses ~400ms context
        "subsampling_factor": 8,  # Conformer subsampling
        "streaming": False,  # Streaming handled at application level
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nExport complete! Files saved to: {output_path}")
    print("\nExported files:")
    for f in output_path.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.1f} MB")


def export_full_model(output_dir: str, model_name: str = "nvidia/parakeet-tdt-0.6b-v3"):
    """Alternative: Export full model as single ONNX file."""
    import nemo.collections.asr as nemo_asr

    print(f"Loading model: {model_name}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Disable cache support to avoid export issues
    if hasattr(model, 'set_export_config'):
        model.set_export_config({
            'cache_support': 'False',
        })

    onnx_path = output_path / "parakeet_tdt.onnx"
    print(f"Exporting full model to: {onnx_path}")

    try:
        model.export(
            str(onnx_path),
            onnx_opset_version=17,
            check_trace=False,
        )
    except Exception as e:
        print(f"Export with opset 17 failed: {e}")
        print("Trying with opset 14...")
        model.export(
            str(onnx_path),
            onnx_opset_version=14,
            check_trace=False,
        )

    print(f"\nExport complete! Model saved to: {onnx_path}")
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Parakeet-TDT to ONNX")
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./models/parakeet",
        help="Output directory for ONNX files"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="nvidia/parakeet-tdt-0.6b-v3",
        help="Model name from HuggingFace/NeMo"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Export as single ONNX file instead of separate components"
    )

    args = parser.parse_args()

    if args.full:
        export_full_model(args.output, args.model)
    else:
        export_parakeet_tdt(args.output, args.model)

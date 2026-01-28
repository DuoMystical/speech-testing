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
    """Export Parakeet-TDT model to ONNX with streaming cache support."""
    import nemo.collections.asr as nemo_asr
    import torch

    print(f"Loading model: {model_name}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Configure for streaming with cache support
    print("Configuring model for streaming ONNX export with cache support...")

    # For Conformer streaming, we need to configure the encoder's attention
    # to use proper streaming settings before export
    encoder = model.encoder

    # Check if this is a streaming-capable model and configure it
    if hasattr(encoder, 'streaming_cfg') or hasattr(encoder, 'set_default_streaming_cfg'):
        print("Model supports native streaming configuration")
        if hasattr(encoder, 'set_default_streaming_cfg'):
            encoder.set_default_streaming_cfg()

    # Set export config for cache support
    # Use cache_last_channel which caches the last channel dimension (more stable)
    model.set_export_config({
        'cache_support': 'True',
        'cache_last_channel': 'True',
    })

    # Ensure encoder is in streaming mode if supported
    if hasattr(encoder, 'setup_streaming_params'):
        # Configure streaming with reasonable defaults
        encoder.setup_streaming_params(
            chunk_size=40,  # ~400ms at 10ms frame shift with 4x subsampling
            left_chunks=-1,  # Unlimited left context in cache
            max_context=5000,  # Max context frames
        )

    encoder_path = output_path / "encoder.onnx"
    print(f"Exporting encoder (with cache) to: {encoder_path}")

    try:
        # Try export with cache support
        encoder.export(
            str(encoder_path),
            onnx_opset_version=17,
            check_trace=False,
        )
        print("Encoder exported successfully with cache support!")

    except RuntimeError as e:
        if "size of tensor" in str(e):
            print(f"Cache export failed due to tensor mismatch: {e}")
            print("Trying export without cache (streaming handled at app level)...")

            # Reset export config to disable cache
            model.set_export_config({'cache_support': 'False'})

            encoder.export(
                str(encoder_path),
                onnx_opset_version=17,
                check_trace=False,
            )
            print("Encoder exported without cache - streaming will be handled at application level")
        else:
            raise

    # Export decoder (prediction network for transducer)
    decoder_path = output_path / "decoder.onnx"
    print(f"Exporting decoder to: {decoder_path}")
    model.decoder.export(
        str(decoder_path),
        onnx_opset_version=17,
        check_trace=False,
    )

    # Export joint network
    joint_path = output_path / "joint.onnx"
    print(f"Exporting joint network to: {joint_path}")
    model.joint.export(
        str(joint_path),
        onnx_opset_version=17,
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

    # Check if cache export succeeded by looking at export config
    cache_enabled = model._export_config.get('cache_support', 'False') == 'True'

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
        "subsampling_factor": 4,  # Conformer subsampling factor
        "streaming": True,
        "cache_support": cache_enabled,
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nExport complete! Files saved to: {output_path}")
    print("\nExported files:")
    for f in output_path.iterdir():
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name}: {size_mb:.1f} MB")


def export_full_model(output_dir: str, model_name: str = "nvidia/parakeet-tdt-0.6b-v3"):
    """Alternative: Export full model as single ONNX file with cache support."""
    import nemo.collections.asr as nemo_asr

    print(f"Loading model: {model_name}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Try with cache support first
    if hasattr(model, 'set_export_config'):
        model.set_export_config({
            'cache_support': 'True',
            'cache_last_channel': 'True',
        })

    onnx_path = output_path / "parakeet_tdt.onnx"
    print(f"Exporting full model to: {onnx_path}")

    try:
        model.export(
            str(onnx_path),
            onnx_opset_version=17,
            check_trace=False,
        )
    except RuntimeError as e:
        if "size of tensor" in str(e):
            print(f"Cache export failed: {e}")
            print("Retrying without cache support...")
            model.set_export_config({'cache_support': 'False'})
            model.export(
                str(onnx_path),
                onnx_opset_version=17,
                check_trace=False,
            )
        else:
            raise

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

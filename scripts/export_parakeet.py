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

    encoder = model.encoder

    # Get encoder configuration
    print("Analyzing encoder configuration...")
    if hasattr(encoder, '_cfg'):
        cfg = encoder._cfg
        print(f"  - d_model: {getattr(cfg, 'd_model', 'unknown')}")
        print(f"  - num_layers: {getattr(cfg, 'num_layers', 'unknown')}")
        print(f"  - subsampling_factor: {getattr(cfg, 'subsampling_factor', 4)}")

    # For Conformer with relative positional encoding, we need to configure
    # the attention context size to match what we'll use during streaming
    # The tensor mismatch occurs when pos_emb size doesn't match the attention window

    # Configure streaming parameters on the encoder BEFORE setting export config
    print("Configuring encoder for streaming...")

    # Set attention context for streaming - this controls the positional embedding size
    # att_context_size is [left_context, right_context] in frames
    # For streaming, we typically use limited left context and 0 right context
    if hasattr(encoder, 'set_streaming_cfg'):
        # Use NeMo's streaming config setter
        encoder.set_streaming_cfg(
            chunk_size=32,  # Process 32 frames at a time (~320ms with 10ms shift)
            left_context_size=32,  # Keep 32 frames of left context
            right_context_size=0,  # No right context (causal/streaming)
        )
        print("  - Streaming config set via set_streaming_cfg")
    elif hasattr(encoder, 'streaming_cfg'):
        # Direct attribute access
        encoder.streaming_cfg = {
            'chunk_size': 32,
            'left_context_size': 32,
            'right_context_size': 0,
        }
        print("  - Streaming config set via streaming_cfg attribute")

    # Reconfigure attention layers for streaming if needed
    if hasattr(encoder, 'layers'):
        for i, layer in enumerate(encoder.layers):
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                # Set attention context size for relative positional encoding
                if hasattr(attn, 'set_streaming'):
                    attn.set_streaming(True)
                # Some models use att_context_size
                if hasattr(attn, 'att_context_size'):
                    attn.att_context_size = [32, 0]  # [left, right] context

    # Now set the export config for cache support
    print("Setting export config with cache support...")
    model.set_export_config({
        'cache_support': 'True',
        'cache_last_channel': 'True',
    })

    # Export encoder with cache
    encoder_path = output_path / "encoder.onnx"
    print(f"Exporting encoder (with cache) to: {encoder_path}")
    encoder.export(
        str(encoder_path),
        onnx_opset_version=17,
        check_trace=False,
    )
    print("Encoder exported successfully with cache support!")

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
        "chunk_size_frames": 32,  # Frames per chunk for streaming
        "left_context_frames": 32,  # Cached left context
        "subsampling_factor": 4,  # Conformer subsampling factor
        "streaming": True,
        "cache_support": True,
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

    # Configure encoder for streaming before export
    encoder = model.encoder
    if hasattr(encoder, 'set_streaming_cfg'):
        encoder.set_streaming_cfg(
            chunk_size=32,
            left_context_size=32,
            right_context_size=0,
        )
    elif hasattr(encoder, 'streaming_cfg'):
        encoder.streaming_cfg = {
            'chunk_size': 32,
            'left_context_size': 32,
            'right_context_size': 0,
        }

    # Configure attention layers
    if hasattr(encoder, 'layers'):
        for layer in encoder.layers:
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                if hasattr(attn, 'set_streaming'):
                    attn.set_streaming(True)
                if hasattr(attn, 'att_context_size'):
                    attn.att_context_size = [32, 0]

    # Set export config with cache support
    model.set_export_config({
        'cache_support': 'True',
        'cache_last_channel': 'True',
    })

    onnx_path = output_path / "parakeet_tdt.onnx"
    print(f"Exporting full model to: {onnx_path}")

    model.export(
        str(onnx_path),
        onnx_opset_version=17,
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

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

    # For FastConformer with relative positional encoding, we need to configure
    # the attention context size to match the cache requirements.
    # The tensor mismatch (10073 vs 5073) occurs because cache mode expects 2x the
    # positional embeddings for [cached + new] context.

    print("Configuring encoder for streaming with cache support...")

    # Streaming chunk configuration
    chunk_size = 32  # frames per chunk
    left_context = 32  # cached left context frames
    right_context = 0  # causal streaming

    # Configure streaming via available methods
    if hasattr(encoder, 'set_streaming_cfg'):
        encoder.set_streaming_cfg(
            chunk_size=chunk_size,
            left_context_size=left_context,
            right_context_size=right_context,
        )
        print(f"  - Streaming config set: chunk={chunk_size}, left={left_context}, right={right_context}")

    # Configure each layer's attention for streaming
    # This is critical for proper positional embedding sizing
    layers_configured = 0
    if hasattr(encoder, 'layers'):
        for layer in encoder.layers:
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                # Enable streaming mode
                if hasattr(attn, 'set_streaming'):
                    attn.set_streaming(True)
                # Set attention context size [left, right]
                if hasattr(attn, 'att_context_size'):
                    attn.att_context_size = [left_context, right_context]
                    layers_configured += 1
                # Some models use streaming_config attribute
                if hasattr(attn, 'streaming_config'):
                    attn.streaming_config = {
                        'chunk_size': chunk_size,
                        'left_context': left_context,
                        'right_context': right_context,
                    }

    if layers_configured > 0:
        print(f"  - Configured {layers_configured} attention layers for streaming")

    # For FastConformer, we may need to adjust the positional encoding
    # to handle the cache size properly
    if hasattr(encoder, 'pos_enc'):
        pos_enc = encoder.pos_enc
        print(f"  - Positional encoding type: {type(pos_enc).__name__}")
        # Check if we need to resize pos encoding for cache
        if hasattr(pos_enc, 'max_len'):
            current_max = pos_enc.max_len
            # Cache mode needs 2x the positional embeddings
            required_max = current_max * 2
            print(f"  - Current max_len: {current_max}, required for cache: {required_max}")
            # Extend positional encodings if needed
            if hasattr(pos_enc, 'extend_pe'):
                pos_enc.extend_pe(required_max)
                print(f"  - Extended positional encodings to {required_max}")

    # Check and configure relative positional bias for attention
    if hasattr(encoder, 'self_attention_model'):
        print(f"  - Self-attention model: {encoder.self_attention_model}")

    # Set the streaming mode flag if available
    if hasattr(encoder, 'streaming'):
        encoder.streaming = True
        print("  - Encoder streaming mode enabled")

    # For FastConformer, we need to ensure the relative positional bias
    # is properly sized for cache-aware export
    # The bias table has size (2 * max_pos - 1) and cache doubles the sequence
    if hasattr(encoder, 'layers'):
        for layer in encoder.layers:
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                # Check for rel_pos_bias in multi-head attention
                if hasattr(attn, 'rel_pos_bias') and attn.rel_pos_bias is not None:
                    rel_pos = attn.rel_pos_bias
                    if hasattr(rel_pos, 'pe') and rel_pos.pe is not None:
                        current_size = rel_pos.pe.size(0)
                        print(f"  - Relative positional bias size: {current_size}")
                        # Extend if needed for cache (2x for cache + current)
                        if hasattr(rel_pos, 'extend_pe'):
                            rel_pos.extend_pe(seq_length=current_size)
                            print(f"  - Extended relative positional bias")

    # For models with local attention, set the chunk length
    if hasattr(encoder, '_cfg') and hasattr(encoder._cfg, 'att_context_size'):
        encoder._cfg.att_context_size = [left_context, right_context]
        print(f"  - Set cfg att_context_size to [{left_context}, {right_context}]")

    # Now set the export config for cache support
    print("Setting export config with cache support...")
    model.set_export_config({
        'cache_support': 'True',
        'cache_last_channel': 'True',
    })

    # Export encoder with cache
    encoder_path = output_path / "encoder.onnx"
    print(f"Exporting encoder (with cache) to: {encoder_path}")

    try:
        encoder.export(
            str(encoder_path),
            onnx_opset_version=17,
            check_trace=False,
        )
        print("Encoder exported successfully with cache support!")
    except RuntimeError as e:
        error_msg = str(e)
        if "shape" in error_msg.lower() or "size" in error_msg.lower():
            print(f"\nCache export failed with tensor mismatch: {e}")
            print("\nAttempting export with adjusted positional encoding...")

            # Try resetting and extending positional encodings
            if hasattr(encoder, 'layers'):
                for layer in encoder.layers:
                    if hasattr(layer, 'self_attn'):
                        attn = layer.self_attn
                        if hasattr(attn, 'rel_pos_bias'):
                            rel_pos = attn.rel_pos_bias
                            # Try to recreate with larger size
                            if hasattr(rel_pos, 'pe') and rel_pos.pe is not None:
                                # Double the positional bias table
                                old_size = rel_pos.pe.size(0)
                                new_size = old_size * 2
                                old_pe = rel_pos.pe
                                # Create new larger tensor
                                new_pe = torch.zeros(new_size, old_pe.size(1), device=old_pe.device, dtype=old_pe.dtype)
                                # Center the old embeddings in new tensor
                                start = (new_size - old_size) // 2
                                new_pe[start:start + old_size] = old_pe
                                rel_pos.pe = torch.nn.Parameter(new_pe, requires_grad=False)
                                print(f"  - Resized positional bias from {old_size} to {new_size}")

            # Retry export
            encoder.export(
                str(encoder_path),
                onnx_opset_version=17,
                check_trace=False,
            )
            print("Encoder exported successfully after adjustment!")
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
    import torch

    print(f"Loading model: {model_name}")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Streaming configuration
    chunk_size = 32
    left_context = 32
    right_context = 0

    # Configure encoder for streaming before export
    encoder = model.encoder
    if hasattr(encoder, 'set_streaming_cfg'):
        encoder.set_streaming_cfg(
            chunk_size=chunk_size,
            left_context_size=left_context,
            right_context_size=right_context,
        )
        print(f"  - Streaming config set: chunk={chunk_size}, left={left_context}, right={right_context}")

    # Configure attention layers
    if hasattr(encoder, 'layers'):
        for layer in encoder.layers:
            if hasattr(layer, 'self_attn'):
                attn = layer.self_attn
                if hasattr(attn, 'set_streaming'):
                    attn.set_streaming(True)
                if hasattr(attn, 'att_context_size'):
                    attn.att_context_size = [left_context, right_context]

    # Extend positional encodings for cache
    if hasattr(encoder, 'pos_enc') and hasattr(encoder.pos_enc, 'extend_pe'):
        if hasattr(encoder.pos_enc, 'max_len'):
            encoder.pos_enc.extend_pe(encoder.pos_enc.max_len * 2)

    # Set export config with cache support
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
        print(f"\nExport complete! Model saved to: {onnx_path}")
    except RuntimeError as e:
        error_msg = str(e)
        if "shape" in error_msg.lower() or "size" in error_msg.lower():
            print(f"\nCache export failed: {e}")
            print("Attempting with adjusted positional encoding...")

            # Resize positional bias for cache
            if hasattr(encoder, 'layers'):
                for layer in encoder.layers:
                    if hasattr(layer, 'self_attn'):
                        attn = layer.self_attn
                        if hasattr(attn, 'rel_pos_bias') and attn.rel_pos_bias is not None:
                            rel_pos = attn.rel_pos_bias
                            if hasattr(rel_pos, 'pe') and rel_pos.pe is not None:
                                old_size = rel_pos.pe.size(0)
                                new_size = old_size * 2
                                old_pe = rel_pos.pe
                                new_pe = torch.zeros(new_size, old_pe.size(1), device=old_pe.device, dtype=old_pe.dtype)
                                start = (new_size - old_size) // 2
                                new_pe[start:start + old_size] = old_pe
                                rel_pos.pe = torch.nn.Parameter(new_pe, requires_grad=False)

            model.export(
                str(onnx_path),
                onnx_opset_version=17,
                check_trace=False,
            )
            print(f"\nExport complete after adjustment! Model saved to: {onnx_path}")
        else:
            raise

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

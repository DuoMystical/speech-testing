#!/bin/bash
set -e

MODELS_DIR="${MODELS_DIR:-/app/models}"
SCRIPTS_DIR="${SCRIPTS_DIR:-/app/scripts}"

echo "=== STT Server Startup ==="
echo "Models directory: $MODELS_DIR"

# Check and download TEN VAD
if [ ! -f "$MODELS_DIR/ten_vad/ten_vad.onnx" ]; then
    echo "TEN VAD model not found, downloading..."
    python3 "$SCRIPTS_DIR/download_ten_vad.py" --output "$MODELS_DIR/ten_vad"
    echo "TEN VAD downloaded successfully"
else
    echo "TEN VAD model found"
fi

# Check and export Parakeet-TDT
if [ ! -f "$MODELS_DIR/parakeet/encoder.onnx" ]; then
    echo "Parakeet-TDT model not found, exporting..."
    python3 "$SCRIPTS_DIR/export_parakeet.py" --output "$MODELS_DIR/parakeet"
    echo "Parakeet-TDT exported successfully"
else
    echo "Parakeet-TDT model found"
fi

echo "=== Starting STT Server ==="
exec stt-server

# Real-time Speech-to-Text Server

A high-performance, real-time speech-to-text server built in Rust using:

- **Parakeet-TDT 0.6B v3** - NVIDIA's state-of-the-art ASR model
- **TEN VAD** - Low-latency voice activity detection
- **ONNX Runtime + TensorRT** - Maximum inference performance on NVIDIA GPUs

## Features

- Real-time streaming transcription
- Voice activity detection (auto start/stop)
- WebSocket-based communication
- Web UI for easy interaction
- Optimized for NVIDIA A100 GPUs with TensorRT FP16

## Quick Start

### 1. Export Models

First, export the models to ONNX format:

```bash
# Install Python dependencies
pip install nemo_toolkit[asr] onnx onnxruntime huggingface_hub

# Export Parakeet-TDT
python scripts/export_parakeet.py --output ./models/parakeet

# Download TEN VAD
python scripts/download_ten_vad.py --output ./models/ten_vad
```

### 2. Build and Run

#### Using Docker (Recommended)

```bash
# Build the image
docker build -t stt-server .

# Run with GPU
docker run --gpus all -p 8080:8080 -v $(pwd)/models:/app/models stt-server
```

#### Using Docker Compose

```bash
docker-compose up
```

#### Local Development

```bash
# Build
cargo build --release

# Run
MODELS_DIR=./models FRONTEND_PATH=./frontend cargo run --release
```

### 3. Access the Web UI

Open http://localhost:8080 in your browser.

## Architecture

```
Browser (Microphone) ──WebSocket──► Rust Server
                                        │
                                        ▼
                                    TEN VAD (ONNX)
                                        │
                                        ▼
                                   Parakeet-TDT (ONNX + TensorRT)
                                        │
                                        ▼
                              ◄──Transcripts──
```

## API

### WebSocket Endpoint: `/ws`

#### Sending Audio

Send binary PCM audio data (16kHz, mono, 16-bit signed little-endian):

```javascript
// Send audio buffer
ws.send(audioBuffer);
```

#### Receiving Messages

```javascript
// VAD state change
{ "type": "vad", "state": "speaking" | "silent" }

// Transcript update
{ "type": "transcript", "text": "...", "is_final": true|false, "confidence": 0.95 }
```

#### Control Messages

```javascript
// Reset session
ws.send(JSON.stringify({ "type": "reset" }));

// Ping for latency measurement
ws.send(JSON.stringify({ "type": "ping" }));
```

### Health Check: `GET /health`

Returns server status and loaded models.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `./models` | Directory containing ONNX models |
| `FRONTEND_PATH` | `./frontend` | Path to web UI files |
| `PORT` | `8080` | Server port |
| `RUST_LOG` | `info` | Log level |
| `ORT_TENSORRT_FP16_ENABLE` | `1` | Enable TensorRT FP16 |
| `ORT_TENSORRT_ENGINE_CACHE_ENABLE` | `1` | Cache TensorRT engines |

## Deployment on Koyeb

1. Push Docker image to your registry
2. Deploy using the Koyeb configuration:

```bash
koyeb deploy -f koyeb.yaml
```

## Performance

On NVIDIA A100 with TensorRT FP16:

| Metric | Value |
|--------|-------|
| VAD Latency | ~2ms per frame |
| ASR Latency | ~20-40ms per chunk |
| End-to-end | ~100-150ms |
| Throughput | Real-time factor ~50x |

## License

MIT

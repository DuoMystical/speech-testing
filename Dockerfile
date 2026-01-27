# Build stage - Ubuntu 24.04 (glibc 2.39) for ort-sys compatibility
FROM ubuntu:24.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install build dependencies and Rust
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust 1.88
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.88.0
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy manifests
COPY Cargo.toml Cargo.lock* ./

# Create dummy main.rs to cache dependencies
RUN mkdir -p src && echo "fn main() {}" > src/main.rs

# Build dependencies only (cached layer)
RUN cargo build --release || true
RUN rm -rf src

# Copy actual source code
COPY src ./src

# Build the actual application
RUN touch src/main.rs && cargo build --release

# Runtime stage - NVIDIA TensorRT base image
FROM nvcr.io/nvidia/tensorrt:24.01-py3

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for model export
RUN pip3 install --no-cache-dir \
    nemo_toolkit[asr] \
    onnx \
    onnxruntime-gpu \
    huggingface_hub

# Find and symlink the ONNX Runtime library for Rust to load dynamically
RUN ONNX_LIB=$(python3 -c "import onnxruntime; import os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi', 'libonnxruntime.so'))") && \
    if [ -f "$ONNX_LIB" ]; then \
        ln -sf "$ONNX_LIB" /usr/local/lib/libonnxruntime.so && \
        ldconfig; \
    fi

# Copy the built binary
COPY --from=builder /app/target/release/stt-server /usr/local/bin/

# Copy frontend
COPY frontend /app/frontend

# Copy scripts (including startup script)
COPY scripts /app/scripts
RUN chmod +x /app/scripts/startup.sh

# Create directories
RUN mkdir -p /app/models/parakeet /app/models/ten_vad /app/tensorrt-cache

# Environment variables
ENV MODELS_DIR=/app/models
ENV SCRIPTS_DIR=/app/scripts
ENV FRONTEND_PATH=/app/frontend
ENV PORT=8080
ENV RUST_LOG=info
ENV ORT_TENSORRT_ENGINE_CACHE_ENABLE=1
ENV ORT_TENSORRT_CACHE_PATH=/app/tensorrt-cache
ENV ORT_TENSORRT_FP16_ENABLE=1
# ONNX Runtime dynamic library path (ort load-dynamic feature)
ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Expose port
EXPOSE 8080

# Health check - longer start period for model download
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run startup script which downloads models then starts server
CMD ["/app/scripts/startup.sh"]

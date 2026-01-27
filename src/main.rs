//! Real-time Speech-to-Text Server
//!
//! WebSocket server for streaming audio transcription using:
//! - TEN VAD for voice activity detection
//! - Parakeet-TDT 0.6B v3 for speech recognition
//! - ONNX Runtime with TensorRT for inference

mod vad;
mod asr;
mod audio;
mod session;
mod pipeline;

use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::IntoResponse,
    routing::get,
    Router,
};
use futures::{sink::SinkExt, stream::StreamExt};
use std::sync::Arc;
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;
use tracing::{error, info, warn};

use pipeline::SttPipeline;
use session::SessionManager;

/// Application state shared across all connections
pub struct AppState {
    session_manager: SessionManager,
    pipeline: Arc<SttPipeline>,
}

impl AppState {
    pub async fn new(models_dir: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let pipeline = SttPipeline::new(models_dir).await?;

        Ok(Self {
            session_manager: SessionManager::new(),
            pipeline: Arc::new(pipeline),
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "stt_server=info,tower_http=info".into()),
        )
        .init();

    // Get models directory from environment
    let models_dir = std::env::var("MODELS_DIR").unwrap_or_else(|_| "./models".to_string());

    info!("Initializing STT pipeline with models from: {}", models_dir);

    let state = match AppState::new(&models_dir).await {
        Ok(state) => Arc::new(RwLock::new(state)),
        Err(e) => {
            error!("Failed to initialize STT pipeline: {}", e);
            error!("Make sure the models are downloaded and exported:");
            error!("  python scripts/export_parakeet.py --output ./models/parakeet");
            error!("  python scripts/download_ten_vad.py --output ./models/ten_vad");
            return Err(e.into());
        }
    };

    // CORS configuration
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Get frontend path from environment
    let frontend_path =
        std::env::var("FRONTEND_PATH").unwrap_or_else(|_| "./frontend".to_string());

    // Build router
    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/health", get(health_handler))
        .nest_service("/", ServeDir::new(&frontend_path))
        .layer(cors)
        .with_state(state);

    // Get port from environment
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);

    let addr = format!("0.0.0.0:{}", port);
    info!("Starting STT Server on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Health check endpoint
async fn health_handler() -> impl IntoResponse {
    axum::Json(serde_json::json!({
        "status": "ok",
        "service": "stt-server",
        "version": env!("CARGO_PKG_VERSION"),
        "models": {
            "vad": "TEN-VAD",
            "asr": "Parakeet-TDT-0.6B-v3"
        }
    }))
}

/// WebSocket upgrade handler
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<RwLock<AppState>>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle an individual WebSocket connection
async fn handle_socket(socket: WebSocket, state: Arc<RwLock<AppState>>) {
    let session_id = uuid::Uuid::new_v4().to_string();
    info!("New WebSocket connection: {}", session_id);

    let (mut sender, mut receiver) = socket.split();

    // Get pipeline reference
    let pipeline = {
        let state_guard = state.read().await;
        state_guard.pipeline.clone()
    };

    // Register session
    {
        let mut state_guard = state.write().await;
        state_guard.session_manager.add_session(&session_id);
    }

    // Create session-specific state
    let session_state = Arc::new(RwLock::new(SessionState::new()));

    // Send welcome message
    let welcome = serde_json::json!({
        "type": "connected",
        "session_id": session_id,
        "message": "Connected to STT Server",
        "config": {
            "sample_rate": 16000,
            "channels": 1,
            "format": "pcm_s16le"
        }
    });
    if sender
        .send(Message::Text(welcome.to_string()))
        .await
        .is_err()
    {
        return;
    }

    // Main message loop
    loop {
        tokio::select! {
            Some(msg) = receiver.next() => {
                match msg {
                    Ok(Message::Binary(data)) => {
                        // Audio data received
                        if let Err(e) = handle_audio_data(
                            &data,
                            &mut sender,
                            &session_id,
                            &pipeline,
                            &session_state,
                        ).await {
                            error!("Error handling audio: {}", e);
                        }
                    }
                    Ok(Message::Text(text)) => {
                        // Control messages
                        if let Err(e) = handle_control_message(
                            &text,
                            &mut sender,
                            &session_state,
                        ).await {
                            error!("Error handling control message: {}", e);
                        }
                    }
                    Ok(Message::Close(_)) => {
                        info!("Client {} requested close", session_id);
                        break;
                    }
                    Ok(Message::Ping(data)) => {
                        let _ = sender.send(Message::Pong(data)).await;
                    }
                    Ok(_) => {}
                    Err(e) => {
                        error!("WebSocket error: {}", e);
                        break;
                    }
                }
            }
            else => break,
        }
    }

    // Cleanup
    {
        let mut state_guard = state.write().await;
        state_guard.session_manager.remove_session(&session_id);
    }
    info!("WebSocket connection closed: {}", session_id);
}

/// Session-specific state
struct SessionState {
    is_speaking: bool,
    audio_buffer: Vec<i16>,
    vad_state: vad::VadState,
}

impl SessionState {
    fn new() -> Self {
        Self {
            is_speaking: false,
            audio_buffer: Vec::new(),
            vad_state: vad::VadState::new(),
        }
    }
}

/// Handle incoming audio data
async fn handle_audio_data(
    data: &[u8],
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    session_id: &str,
    pipeline: &Arc<SttPipeline>,
    session_state: &Arc<RwLock<SessionState>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Convert bytes to i16 samples (assuming PCM S16LE)
    let samples: Vec<i16> = data
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    if samples.is_empty() {
        return Ok(());
    }

    // Process through pipeline
    let events = pipeline.process_audio(&samples, session_state).await?;

    // Send events to client
    for event in events {
        let msg = match event {
            pipeline::SttEvent::VadStart => {
                serde_json::json!({
                    "type": "vad",
                    "state": "speaking"
                })
            }
            pipeline::SttEvent::VadEnd => {
                serde_json::json!({
                    "type": "vad",
                    "state": "silent"
                })
            }
            pipeline::SttEvent::PartialTranscript { text, is_final } => {
                serde_json::json!({
                    "type": "transcript",
                    "text": text,
                    "is_final": is_final
                })
            }
            pipeline::SttEvent::FinalTranscript { text, confidence } => {
                serde_json::json!({
                    "type": "transcript",
                    "text": text,
                    "is_final": true,
                    "confidence": confidence
                })
            }
        };

        sender.send(Message::Text(msg.to_string())).await?;
    }

    Ok(())
}

/// Handle control messages (JSON)
async fn handle_control_message(
    text: &str,
    sender: &mut futures::stream::SplitSink<WebSocket, Message>,
    session_state: &Arc<RwLock<SessionState>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let msg: serde_json::Value = serde_json::from_str(text)?;

    match msg.get("type").and_then(|t| t.as_str()) {
        Some("reset") => {
            // Reset session state
            let mut state = session_state.write().await;
            state.is_speaking = false;
            state.audio_buffer.clear();
            state.vad_state = vad::VadState::new();

            let response = serde_json::json!({
                "type": "reset",
                "message": "Session reset"
            });
            sender.send(Message::Text(response.to_string())).await?;
        }
        Some("ping") => {
            let response = serde_json::json!({
                "type": "pong",
                "timestamp": chrono_timestamp()
            });
            sender.send(Message::Text(response.to_string())).await?;
        }
        _ => {
            warn!("Unknown control message type: {:?}", msg.get("type"));
        }
    }

    Ok(())
}

fn chrono_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

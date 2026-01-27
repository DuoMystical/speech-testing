//! STT Pipeline: VAD -> ASR
//!
//! Orchestrates voice activity detection and speech recognition.

use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::asr::{AsrConfig, AsrState, ParakeetAsr};
use crate::audio::pcm_i16_to_f32;
use crate::vad::{TenVad, VadConfig, VadEvent};
use crate::SessionState;

/// Events emitted by the STT pipeline
#[derive(Debug, Clone)]
pub enum SttEvent {
    /// Voice activity started
    VadStart,
    /// Voice activity ended
    VadEnd,
    /// Partial transcript update
    PartialTranscript { text: String, is_final: bool },
    /// Final transcript after speech ended
    FinalTranscript { text: String, confidence: f32 },
}

/// Configuration for the STT pipeline
pub struct PipelineConfig {
    pub vad: VadConfig,
    pub asr: AsrConfig,
    /// Minimum speech duration in ms to trigger transcription
    pub min_speech_ms: u64,
    /// Maximum speech duration in ms before forcing transcription
    pub max_speech_ms: u64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            vad: VadConfig::default(),
            asr: AsrConfig::default(),
            min_speech_ms: 200,
            max_speech_ms: 30000, // 30 seconds max
        }
    }
}

/// Complete STT pipeline combining VAD and ASR
pub struct SttPipeline {
    vad: Mutex<TenVad>,
    asr: Mutex<ParakeetAsr>,
    config: PipelineConfig,
}

impl SttPipeline {
    /// Create new STT pipeline
    pub async fn new(models_dir: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let models_path = Path::new(models_dir);
        let config = PipelineConfig::default();

        // Load VAD model
        let vad_model_path = models_path.join("ten_vad").join("ten_vad.onnx");
        info!("Loading VAD model from: {:?}", vad_model_path);

        let vad = if vad_model_path.exists() {
            TenVad::new(&vad_model_path, config.vad.clone())?
        } else {
            // Return error if model not found
            return Err(format!(
                "VAD model not found at {:?}. Run: python scripts/download_ten_vad.py",
                vad_model_path
            )
            .into());
        };

        // Load ASR model
        let asr_model_dir = models_path.join("parakeet");
        info!("Loading ASR model from: {:?}", asr_model_dir);

        let asr = if asr_model_dir.exists() {
            ParakeetAsr::new(&asr_model_dir, config.asr.clone())?
        } else {
            return Err(format!(
                "ASR model not found at {:?}. Run: python scripts/export_parakeet.py",
                asr_model_dir
            )
            .into());
        };

        info!("STT pipeline initialized successfully");

        Ok(Self {
            vad: Mutex::new(vad),
            asr: Mutex::new(asr),
            config,
        })
    }

    /// Process audio samples through the pipeline
    pub async fn process_audio(
        &self,
        samples: &[i16],
        session_state: &Arc<RwLock<SessionState>>,
    ) -> Result<Vec<SttEvent>, Box<dyn std::error::Error + Send + Sync>> {
        let mut events = Vec::new();

        // Convert to f32
        let samples_f32 = pcm_i16_to_f32(samples);

        // Get current state
        let mut state = session_state.write().await;

        // Process through VAD
        let vad_events = self.vad.lock().unwrap().process_audio(&samples_f32, &mut state.vad_state)?;

        for vad_event in vad_events {
            match vad_event {
                VadEvent::SpeechStart => {
                    if !state.is_speaking {
                        state.is_speaking = true;
                        state.audio_buffer.clear();
                        events.push(SttEvent::VadStart);
                        debug!("Speech started");
                    }
                }
                VadEvent::SpeechEnd => {
                    if state.is_speaking {
                        state.is_speaking = false;
                        events.push(SttEvent::VadEnd);
                        debug!("Speech ended, buffer size: {} samples", state.audio_buffer.len());

                        // Finalize transcription if we have audio
                        if !state.audio_buffer.is_empty() {
                            // Convert buffer to f32 and process
                            let audio_f32 = pcm_i16_to_f32(&state.audio_buffer);

                            // Create temporary ASR state for final processing
                            let mut asr_state = AsrState::new();
                            let mut asr = self.asr.lock().unwrap();
                            let _ = asr.process_chunk(&audio_f32, &mut asr_state)?;
                            let result = asr.finalize(&mut asr_state)?;

                            if !result.text.is_empty() {
                                events.push(SttEvent::FinalTranscript {
                                    text: result.text,
                                    confidence: result.confidence,
                                });
                            }

                            state.audio_buffer.clear();
                        }
                    }
                }
                VadEvent::Speaking => {
                    // Add samples to buffer while speaking
                    if state.is_speaking {
                        state.audio_buffer.extend_from_slice(samples);

                        // Periodic transcription for long utterances
                        let buffer_ms = state.audio_buffer.len() as u64 * 1000 / 16000;
                        if buffer_ms >= 2000 && buffer_ms % 1000 < 100 {
                            // Every ~1 second, emit partial
                            let audio_f32 = pcm_i16_to_f32(&state.audio_buffer);
                            let mut asr_state = AsrState::new();
                            let result = self.asr.lock().unwrap().process_chunk(&audio_f32, &mut asr_state)?;

                            if !result.text.is_empty() {
                                events.push(SttEvent::PartialTranscript {
                                    text: result.text,
                                    is_final: false,
                                });
                            }
                        }
                    }
                }
                VadEvent::Silent => {
                    // Nothing to do during silence
                }
            }
        }

        // Add samples to buffer if speaking
        if state.is_speaking && !samples.is_empty() {
            // Samples already added in Speaking event handler
        }

        Ok(events)
    }

    /// Get VAD frame size in samples
    pub fn vad_frame_size(&self) -> usize {
        self.vad.lock().unwrap().frame_size()
    }

    /// Get sample rate
    pub fn sample_rate(&self) -> u32 {
        self.vad.lock().unwrap().sample_rate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_defaults() {
        let config = PipelineConfig::default();
        assert_eq!(config.min_speech_ms, 200);
        assert_eq!(config.max_speech_ms, 30000);
    }
}

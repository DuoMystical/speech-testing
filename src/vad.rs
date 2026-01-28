//! TEN VAD (Voice Activity Detection) integration
//!
//! Uses Silero VAD ONNX model for low-latency speech detection.
//! Silero VAD is a lightweight, fast, and accurate voice activity detector.

use ort::{session::Session, value::Tensor};
use std::path::Path;
use tracing::{debug, info};

/// VAD configuration
#[derive(Clone)]
pub struct VadConfig {
    /// Speech probability threshold (0.0 - 1.0)
    pub threshold: f32,
    /// Number of consecutive speech frames to trigger speech start
    pub min_speech_frames: usize,
    /// Number of consecutive silence frames to trigger speech end
    pub min_silence_frames: usize,
    /// Sample rate (must be 16000 for Silero VAD)
    pub sample_rate: u32,
    /// Frame size in samples (512 samples = 32ms at 16kHz for Silero VAD)
    pub frame_size: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_frames: 2,    // ~64ms of speech to start
            min_silence_frames: 10,  // ~320ms of silence to end
            sample_rate: 16000,
            frame_size: 512, // 32ms at 16kHz (Silero VAD requirement)
        }
    }
}

/// VAD state machine
#[derive(Debug, Clone)]
pub struct VadState {
    /// Whether currently in speech
    pub is_speaking: bool,
    /// Consecutive speech frames counter
    speech_frames: usize,
    /// Consecutive silence frames counter
    silence_frames: usize,
    /// Internal state for Silero VAD (2, 1, 128) tensor flattened
    /// Shape: [2, 1, 128] = 256 floats for h and c states combined
    state: Vec<f32>,
}

impl VadState {
    pub fn new() -> Self {
        // Silero VAD uses state tensor of shape (2, 1, 128)
        // First dimension is 2 (h and c states), second is batch=1, third is hidden_size=128
        let state_size = 2 * 1 * 128;
        Self {
            is_speaking: false,
            speech_frames: 0,
            silence_frames: 0,
            state: vec![0.0; state_size],
        }
    }

    pub fn reset(&mut self) {
        self.is_speaking = false;
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.state.fill(0.0);
    }
}

impl Default for VadState {
    fn default() -> Self {
        Self::new()
    }
}

/// VAD detection result
#[derive(Debug, Clone)]
pub enum VadEvent {
    /// Speech started
    SpeechStart,
    /// Speech ended
    SpeechEnd,
    /// Continuing speech
    Speaking,
    /// Continuing silence
    Silent,
}

/// Silero VAD model wrapper (TEN VAD compatible interface)
pub struct TenVad {
    session: Session,
    config: VadConfig,
}

impl TenVad {
    /// Load Silero VAD model from ONNX file
    pub fn new(model_path: impl AsRef<Path>, config: VadConfig) -> Result<Self, ort::Error> {
        let model_path = model_path.as_ref();
        info!("Loading Silero VAD model from: {:?}", model_path);

        // Create ONNX Runtime session with TensorRT EP
        let session = Session::builder()?
            .with_execution_providers([
                ort::ep::TensorRT::default()
                    .with_fp16(true)
                    .build(),
                ort::ep::CUDA::default().build(),
                ort::ep::CPU::default().build(),
            ])?
            .commit_from_file(model_path)?;

        info!("Silero VAD model loaded successfully");
        Ok(Self { session, config })
    }

    /// Process a single frame of audio (32ms = 512 samples at 16kHz for Silero VAD)
    pub fn process_frame(
        &mut self,
        samples: &[f32],
        state: &mut VadState,
    ) -> Result<VadEvent, ort::Error> {
        assert_eq!(
            samples.len(),
            self.config.frame_size,
            "Frame size must be {} samples",
            self.config.frame_size
        );

        // Prepare inputs as tensors for Silero VAD
        // Silero VAD expects:
        //   - input: (batch, audio_len) - audio samples
        //   - sr: scalar int64 - sample rate (16000)
        //   - state: (2, batch, 128) - LSTM h and c states
        let input_data: Vec<f32> = samples.to_vec();
        let input = Tensor::from_array(([1usize, samples.len()], input_data))?;

        // Sample rate as i64
        let sr = Tensor::from_array(([1usize], vec![self.config.sample_rate as i64]))?;

        // State tensor: shape (2, 1, 128)
        let state_data: Vec<f32> = state.state.clone();
        let state_tensor = Tensor::from_array(([2usize, 1usize, 128usize], state_data))?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "input" => input,
            "sr" => sr,
            "state" => state_tensor,
        ])?;

        // Extract outputs
        // Silero VAD outputs:
        //   - output: (batch, 1) - speech probability
        //   - stateN: (2, batch, 128) - new LSTM state
        let (_, prob_data) = outputs["output"].try_extract_tensor::<f32>()?;
        let prob: f32 = prob_data.first().copied().unwrap_or(0.0);

        // Update LSTM state
        let (_, new_state) = outputs["stateN"].try_extract_tensor::<f32>()?;
        state.state.copy_from_slice(new_state);

        // Update state machine
        let is_speech = prob > self.config.threshold;

        let event = if is_speech {
            state.silence_frames = 0;
            state.speech_frames += 1;

            if !state.is_speaking && state.speech_frames >= self.config.min_speech_frames {
                state.is_speaking = true;
                debug!("VAD: Speech started (prob={:.3})", prob);
                VadEvent::SpeechStart
            } else if state.is_speaking {
                VadEvent::Speaking
            } else {
                VadEvent::Silent
            }
        } else {
            state.speech_frames = 0;
            state.silence_frames += 1;

            if state.is_speaking && state.silence_frames >= self.config.min_silence_frames {
                state.is_speaking = false;
                debug!("VAD: Speech ended (silence frames={})", state.silence_frames);
                VadEvent::SpeechEnd
            } else if state.is_speaking {
                VadEvent::Speaking
            } else {
                VadEvent::Silent
            }
        };

        Ok(event)
    }

    /// Process multiple frames of audio
    pub fn process_audio(
        &mut self,
        samples: &[f32],
        state: &mut VadState,
    ) -> Result<Vec<VadEvent>, ort::Error> {
        let mut events = Vec::new();

        for frame in samples.chunks(self.config.frame_size) {
            if frame.len() == self.config.frame_size {
                let event = self.process_frame(frame, state)?;
                events.push(event);
            }
        }

        Ok(events)
    }

    /// Get the frame size in samples
    pub fn frame_size(&self) -> usize {
        self.config.frame_size
    }

    /// Get the sample rate
    pub fn sample_rate(&self) -> u32 {
        self.config.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vad_state_machine() {
        let mut state = VadState::new();
        let config = VadConfig::default();

        // Verify state size is correct for Silero VAD
        assert_eq!(state.state.len(), 2 * 1 * 128);

        // Simulate speech frames
        for _ in 0..5 {
            state.speech_frames += 1;
            state.silence_frames = 0;
        }

        assert!(state.speech_frames >= config.min_speech_frames);
    }

    #[test]
    fn test_vad_config_defaults() {
        let config = VadConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.frame_size, 512); // 32ms for Silero VAD
        assert_eq!(config.threshold, 0.5);
    }
}

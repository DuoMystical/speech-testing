//! TEN VAD (Voice Activity Detection) integration
//!
//! Uses the TEN VAD ONNX model for low-latency speech detection.

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
    /// Sample rate (must be 16000 for TEN VAD)
    pub sample_rate: u32,
    /// Frame size in samples (160 = 10ms at 16kHz)
    pub frame_size: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_frames: 3,    // ~30ms of speech to start
            min_silence_frames: 30,  // ~300ms of silence to end
            sample_rate: 16000,
            frame_size: 160, // 10ms at 16kHz
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
    /// Internal state for TEN VAD (hidden state from previous frame)
    h_state: Vec<f32>,
    c_state: Vec<f32>,
}

impl VadState {
    pub fn new() -> Self {
        // TEN VAD uses LSTM with hidden size of 64
        let hidden_size = 64;
        Self {
            is_speaking: false,
            speech_frames: 0,
            silence_frames: 0,
            h_state: vec![0.0; hidden_size],
            c_state: vec![0.0; hidden_size],
        }
    }

    pub fn reset(&mut self) {
        self.is_speaking = false;
        self.speech_frames = 0;
        self.silence_frames = 0;
        self.h_state.fill(0.0);
        self.c_state.fill(0.0);
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

/// TEN VAD model wrapper
pub struct TenVad {
    session: Session,
    config: VadConfig,
}

impl TenVad {
    /// Load TEN VAD model from ONNX file
    pub fn new(model_path: impl AsRef<Path>, config: VadConfig) -> Result<Self, ort::Error> {
        let model_path = model_path.as_ref();
        info!("Loading TEN VAD model from: {:?}", model_path);

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

        info!("TEN VAD model loaded successfully");
        Ok(Self { session, config })
    }

    /// Process a single frame of audio (10ms = 160 samples at 16kHz)
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

        // Prepare inputs as tensors
        // TEN VAD expects: input (1, frame_size), h (1, hidden_size), c (1, hidden_size)
        let input_data: Vec<f32> = samples.to_vec();
        let input = Tensor::from_array(([1usize, samples.len()], input_data))?;

        let h_data: Vec<f32> = state.h_state.clone();
        let h_in = Tensor::from_array(([1usize, state.h_state.len()], h_data))?;

        let c_data: Vec<f32> = state.c_state.clone();
        let c_in = Tensor::from_array(([1usize, state.c_state.len()], c_data))?;

        // Run inference
        let outputs = self.session.run(ort::inputs![
            "input" => input,
            "h" => h_in,
            "c" => c_in,
        ])?;

        // Extract outputs
        // TEN VAD outputs: prob (1,), h_out (1, hidden_size), c_out (1, hidden_size)
        let (_, prob_data) = outputs["prob"].try_extract_tensor::<f32>()?;
        let prob: f32 = prob_data.first().copied().unwrap_or(0.0);

        // Update LSTM state
        let (_, h_out) = outputs["h_out"].try_extract_tensor::<f32>()?;
        let (_, c_out) = outputs["c_out"].try_extract_tensor::<f32>()?;

        state.h_state.copy_from_slice(h_out);
        state.c_state.copy_from_slice(c_out);

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

        // Simulate speech frames
        for _ in 0..5 {
            state.speech_frames += 1;
            state.silence_frames = 0;
        }

        assert!(state.speech_frames >= 3);
    }
}

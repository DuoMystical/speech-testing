//! Parakeet-TDT ASR (Automatic Speech Recognition) integration
//!
//! Streaming speech recognition using Parakeet-TDT 0.6B v3 with ONNX Runtime.

use ort::{inputs, Session, SessionOutputs};
use std::path::Path;
use tracing::{debug, info};

/// ASR configuration
#[derive(Clone)]
pub struct AsrConfig {
    /// Sample rate (must be 16000)
    pub sample_rate: u32,
    /// Number of mel filterbanks
    pub n_mels: usize,
    /// Frame length in milliseconds
    pub frame_length_ms: f32,
    /// Frame shift/hop in milliseconds
    pub frame_shift_ms: f32,
    /// Chunk size for streaming (in frames)
    pub chunk_size_frames: usize,
    /// Left context for streaming (in frames)
    pub left_context_frames: usize,
    /// Right context for streaming (in frames)
    pub right_context_frames: usize,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_mels: 80,
            frame_length_ms: 25.0,
            frame_shift_ms: 10.0,
            chunk_size_frames: 140,     // ~1.4 seconds
            left_context_frames: 70,
            right_context_frames: 13,
        }
    }
}

/// Streaming ASR state
pub struct AsrState {
    /// Cached encoder states for streaming
    cache: Option<EncoderCache>,
    /// Accumulated transcription
    pub transcript: String,
    /// Audio buffer for incomplete frames
    audio_buffer: Vec<f32>,
    /// Feature buffer for incomplete chunks
    feature_buffer: Vec<Vec<f32>>,
}

impl AsrState {
    pub fn new() -> Self {
        Self {
            cache: None,
            transcript: String::new(),
            audio_buffer: Vec::new(),
            feature_buffer: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.cache = None;
        self.transcript.clear();
        self.audio_buffer.clear();
        self.feature_buffer.clear();
    }
}

impl Default for AsrState {
    fn default() -> Self {
        Self::new()
    }
}

/// Encoder cache for streaming inference
struct EncoderCache {
    /// Cached key-value states from attention layers
    cache_states: Vec<f32>,
    /// Number of processed frames
    processed_frames: usize,
}

/// Transcription result
#[derive(Debug, Clone)]
pub struct TranscriptResult {
    /// Transcribed text
    pub text: String,
    /// Whether this is a final result
    pub is_final: bool,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

/// Parakeet-TDT ASR model
pub struct ParakeetAsr {
    /// Encoder session
    encoder: Session,
    /// Decoder session
    decoder: Session,
    /// Joint network session
    joint: Session,
    /// Vocabulary for token decoding
    vocab: Vec<String>,
    /// Configuration
    config: AsrConfig,
}

impl ParakeetAsr {
    /// Load Parakeet-TDT model from directory containing ONNX files
    pub fn new(model_dir: impl AsRef<Path>, config: AsrConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let model_dir = model_dir.as_ref();
        info!("Loading Parakeet-TDT ASR from: {:?}", model_dir);

        // Helper to create execution providers (can't clone EPs)
        let trt_cache_path = model_dir.join("trt_cache").to_string_lossy().to_string();
        let create_eps = || [
            ort::TensorRTExecutionProvider::default()
                .with_fp16(true)
                .with_engine_cache_enable(true)
                .with_engine_cache_path(trt_cache_path.clone())
                .build(),
            ort::CUDAExecutionProvider::default().build(),
            ort::CPUExecutionProvider::default().build(),
        ];

        // Load encoder
        let encoder_path = model_dir.join("encoder.onnx");
        info!("Loading encoder from: {:?}", encoder_path);
        let encoder = Session::builder()?
            .with_execution_providers(create_eps())?
            .commit_from_file(&encoder_path)?;

        // Load decoder
        let decoder_path = model_dir.join("decoder.onnx");
        info!("Loading decoder from: {:?}", decoder_path);
        let decoder = Session::builder()?
            .with_execution_providers(create_eps())?
            .commit_from_file(&decoder_path)?;

        // Load joint network
        let joint_path = model_dir.join("joint.onnx");
        info!("Loading joint network from: {:?}", joint_path);
        let joint = Session::builder()?
            .with_execution_providers(create_eps())?
            .commit_from_file(&joint_path)?;

        // Load vocabulary
        let vocab_path = model_dir.join("vocab.txt");
        let vocab = if vocab_path.exists() {
            std::fs::read_to_string(&vocab_path)?
                .lines()
                .map(|s| s.to_string())
                .collect()
        } else {
            // Use default BPE vocab placeholder
            vec!["<blank>".to_string(), "<unk>".to_string()]
        };

        info!(
            "Parakeet-TDT loaded: {} vocab tokens",
            vocab.len()
        );

        Ok(Self {
            encoder,
            decoder,
            joint,
            vocab,
            config,
        })
    }

    /// Extract mel spectrogram features from audio
    fn extract_features(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        // Simple mel spectrogram extraction
        // In production, use a proper mel spectrogram library
        let frame_length = (self.config.frame_length_ms * self.config.sample_rate as f32 / 1000.0) as usize;
        let frame_shift = (self.config.frame_shift_ms * self.config.sample_rate as f32 / 1000.0) as usize;

        let mut features = Vec::new();

        for start in (0..samples.len().saturating_sub(frame_length)).step_by(frame_shift) {
            let frame = &samples[start..start + frame_length];

            // Apply Hamming window
            let windowed: Vec<f32> = frame
                .iter()
                .enumerate()
                .map(|(i, &s)| {
                    let window = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (frame_length - 1) as f32).cos();
                    s * window
                })
                .collect();

            // Simple power spectrum (placeholder - should use FFT + mel filterbank)
            // For now, just compute frame energy as placeholder
            let energy: f32 = windowed.iter().map(|x| x * x).sum::<f32>().sqrt();

            // Create mel feature vector (placeholder)
            let mel_features: Vec<f32> = (0..self.config.n_mels)
                .map(|i| {
                    // Placeholder: distribute energy across mel bins
                    let scale = 1.0 - (i as f32 / self.config.n_mels as f32);
                    (energy * scale).ln().max(-10.0)
                })
                .collect();

            features.push(mel_features);
        }

        features
    }

    /// Process audio chunk for streaming transcription
    pub fn process_chunk(
        &self,
        samples: &[f32],
        state: &mut AsrState,
    ) -> Result<TranscriptResult, ort::Error> {
        // Add samples to buffer
        state.audio_buffer.extend_from_slice(samples);

        // Extract features from buffered audio
        let new_features = self.extract_features(&state.audio_buffer);

        if new_features.is_empty() {
            return Ok(TranscriptResult {
                text: state.transcript.clone(),
                is_final: false,
                confidence: 0.0,
            });
        }

        // Add to feature buffer
        state.feature_buffer.extend(new_features);

        // Clear processed audio (keep some for overlap)
        let keep_samples = (self.config.frame_length_ms * self.config.sample_rate as f32 / 1000.0) as usize;
        if state.audio_buffer.len() > keep_samples {
            state.audio_buffer.drain(..state.audio_buffer.len() - keep_samples);
        }

        // Check if we have enough frames for a chunk
        if state.feature_buffer.len() < self.config.chunk_size_frames {
            return Ok(TranscriptResult {
                text: state.transcript.clone(),
                is_final: false,
                confidence: 0.0,
            });
        }

        // Process chunk through encoder
        let chunk_features: Vec<f32> = state.feature_buffer
            .iter()
            .take(self.config.chunk_size_frames)
            .flatten()
            .copied()
            .collect();

        // Remove processed frames (keep context)
        let keep_frames = self.config.left_context_frames;
        if state.feature_buffer.len() > keep_frames {
            state.feature_buffer.drain(..state.feature_buffer.len() - keep_frames);
        }

        // Prepare encoder input
        let batch_size = 1;
        let seq_len = self.config.chunk_size_frames;
        let feature_dim = self.config.n_mels;

        let encoder_input = ndarray::Array3::from_shape_vec(
            (batch_size, seq_len, feature_dim),
            chunk_features,
        ).expect("Failed to create encoder input");

        let input_lengths = ndarray::Array1::from_vec(vec![seq_len as i64]);

        // Run encoder
        let encoder_outputs: SessionOutputs = self.encoder.run(inputs![
            "audio_signal" => encoder_input,
            "length" => input_lengths,
        ]?)?;

        // Get encoder output
        let encoder_out = encoder_outputs["outputs"]
            .try_extract_tensor::<f32>()?;

        // Run greedy decoding with decoder and joint
        let decoded_tokens = self.greedy_decode(&encoder_out)?;

        // Convert tokens to text
        let text = self.tokens_to_text(&decoded_tokens);

        // Update transcript
        if !text.is_empty() {
            if !state.transcript.is_empty() && !state.transcript.ends_with(' ') {
                state.transcript.push(' ');
            }
            state.transcript.push_str(&text);
        }

        debug!("Partial transcript: {}", state.transcript);

        Ok(TranscriptResult {
            text: state.transcript.clone(),
            is_final: false,
            confidence: 0.8, // Placeholder confidence
        })
    }

    /// Finalize transcription (process remaining audio)
    pub fn finalize(&self, state: &mut AsrState) -> Result<TranscriptResult, ort::Error> {
        // Process any remaining audio in buffer
        if !state.audio_buffer.is_empty() {
            // Pad to minimum length if needed
            let min_samples = (self.config.frame_length_ms * self.config.sample_rate as f32 / 1000.0) as usize * 10;
            while state.audio_buffer.len() < min_samples {
                state.audio_buffer.push(0.0);
            }

            // Process final chunk
            let _ = self.process_chunk(&[], state)?;
        }

        let result = TranscriptResult {
            text: state.transcript.clone(),
            is_final: true,
            confidence: 0.9,
        };

        // Reset state
        state.reset();

        Ok(result)
    }

    /// Greedy decoding using decoder and joint network
    fn greedy_decode(
        &self,
        encoder_out: &ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<ndarray::IxDynImpl>>,
    ) -> Result<Vec<usize>, ort::Error> {
        let mut tokens = Vec::new();
        let blank_id = 0;

        // Get encoder output shape
        let seq_len = encoder_out.shape()[1];

        // Initialize decoder state
        let mut decoder_state = vec![0.0f32; 512]; // Placeholder state size

        for t in 0..seq_len {
            // Get encoder output at time t
            let enc_t: Vec<f32> = (0..encoder_out.shape()[2])
                .map(|i| encoder_out[[0, t, i]])
                .collect();

            // Run decoder (simplified - actual implementation would maintain state)
            let dec_input = if tokens.is_empty() {
                vec![blank_id as i64]
            } else {
                vec![*tokens.last().unwrap() as i64]
            };

            let dec_input_arr = ndarray::Array2::from_shape_vec((1, 1), dec_input)
                .expect("Failed to create decoder input");

            let decoder_outputs: SessionOutputs = self.decoder.run(inputs![
                "targets" => dec_input_arr,
            ]?)?;

            let dec_out = decoder_outputs["outputs"]
                .try_extract_tensor::<f32>()?;

            // Run joint network
            let enc_arr = ndarray::Array2::from_shape_vec((1, enc_t.len()), enc_t)
                .expect("Failed to create enc array");

            let dec_vec: Vec<f32> = dec_out.view().iter().copied().collect();
            let dec_arr = ndarray::Array2::from_shape_vec((1, dec_vec.len()), dec_vec)
                .expect("Failed to create dec array");

            let joint_outputs: SessionOutputs = self.joint.run(inputs![
                "encoder_outputs" => enc_arr,
                "decoder_outputs" => dec_arr,
            ]?)?;

            let logits = joint_outputs["outputs"]
                .try_extract_tensor::<f32>()?;

            // Find argmax token
            let logits_slice = logits.view();
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;

            for (i, &v) in logits_slice.iter().enumerate() {
                if v > max_val {
                    max_val = v;
                    max_idx = i;
                }
            }

            // If not blank, add to output
            if max_idx != blank_id {
                tokens.push(max_idx);
            }
        }

        Ok(tokens)
    }

    /// Convert token IDs to text
    fn tokens_to_text(&self, tokens: &[usize]) -> String {
        tokens
            .iter()
            .filter_map(|&id| self.vocab.get(id))
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join("")
            .replace("▁", " ")
            .trim()
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_asr_state() {
        let mut state = AsrState::new();
        state.transcript = "Hello".to_string();
        state.reset();
        assert!(state.transcript.is_empty());
    }
}

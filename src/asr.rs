//! Parakeet-TDT ASR (Automatic Speech Recognition) integration
//!
//! Streaming speech recognition using Parakeet-TDT 0.6B v3 with ONNX Runtime.

use ort::{session::Session, value::Tensor};
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
            transcript: String::new(),
            audio_buffer: Vec::new(),
            feature_buffer: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
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
            ort::ep::TensorRT::default()
                .with_fp16(true)
                .with_engine_cache(true)
                .with_engine_cache_path(&trt_cache_path)
                .build(),
            ort::ep::CUDA::default().build(),
            ort::ep::CPU::default().build(),
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
        &mut self,
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

        // Prepare encoder input as tensor
        let batch_size: usize = 1;
        let seq_len: usize = self.config.chunk_size_frames;
        let feature_dim: usize = self.config.n_mels;

        let encoder_input = Tensor::from_array(([batch_size, seq_len, feature_dim], chunk_features))?;
        let input_lengths = Tensor::from_array(([1usize], vec![seq_len as i64]))?;

        // Run encoder
        let encoder_outputs = self.encoder.run(ort::inputs![
            "audio_signal" => encoder_input,
            "length" => input_lengths,
        ])?;

        // Get encoder output - copy data to owned types to release borrow
        let (encoder_shape, encoder_data) = encoder_outputs["outputs"]
            .try_extract_tensor::<f32>()?;
        let encoder_shape_owned: Vec<i64> = encoder_shape.iter().copied().collect();
        let encoder_data_owned: Vec<f32> = encoder_data.to_vec();
        drop(encoder_outputs);

        // Run greedy decoding with decoder and joint
        let decoded_tokens = self.greedy_decode(&encoder_shape_owned, &encoder_data_owned)?;

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
    pub fn finalize(&mut self, state: &mut AsrState) -> Result<TranscriptResult, ort::Error> {
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
        &mut self,
        encoder_shape: &[i64],
        encoder_data: &[f32],
    ) -> Result<Vec<usize>, ort::Error> {
        let mut tokens = Vec::new();
        let blank_id = 0;

        // Get encoder output shape [batch, seq_len, enc_dim]
        let seq_len = encoder_shape[1] as usize;
        let enc_dim = encoder_shape[2] as usize;

        for t in 0..seq_len {
            // Get encoder output at time t (index into flat array)
            // For shape [1, seq_len, enc_dim], index [0, t, i] = t * enc_dim + i
            let enc_t: Vec<f32> = (0..enc_dim)
                .map(|i| encoder_data[t * enc_dim + i])
                .collect();

            // Run decoder (simplified - actual implementation would maintain state)
            let dec_input = if tokens.is_empty() {
                vec![blank_id as i64]
            } else {
                vec![*tokens.last().unwrap() as i64]
            };

            let dec_input_tensor = Tensor::from_array(([1usize, 1usize], dec_input))?;

            let decoder_outputs = self.decoder.run(ort::inputs![
                "targets" => dec_input_tensor,
            ])?;

            let (_, dec_out) = decoder_outputs["outputs"]
                .try_extract_tensor::<f32>()?;

            // Run joint network
            let enc_tensor = Tensor::from_array(([1usize, enc_t.len()], enc_t))?;

            let dec_vec: Vec<f32> = dec_out.iter().copied().collect();
            let dec_tensor = Tensor::from_array(([1usize, dec_vec.len()], dec_vec))?;

            let joint_outputs = self.joint.run(ort::inputs![
                "encoder_outputs" => enc_tensor,
                "decoder_outputs" => dec_tensor,
            ])?;

            let (_, logits) = joint_outputs["outputs"]
                .try_extract_tensor::<f32>()?;

            // Find argmax token
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;

            for (i, &v) in logits.iter().enumerate() {
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

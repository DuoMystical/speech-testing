//! Audio processing utilities
//!
//! Resampling, format conversion, and mel spectrogram extraction.

use tracing::debug;

/// Audio format configuration
pub struct AudioConfig {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo)
    pub channels: u16,
    /// Bits per sample (16 for PCM S16LE)
    pub bits_per_sample: u16,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            bits_per_sample: 16,
        }
    }
}

/// Convert i16 PCM samples to f32 normalized [-1.0, 1.0]
pub fn pcm_i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples
        .iter()
        .map(|&s| s as f32 / i16::MAX as f32)
        .collect()
}

/// Convert f32 normalized samples to i16 PCM
pub fn pcm_f32_to_i16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect()
}

/// Convert bytes to i16 samples (little-endian)
pub fn bytes_to_i16_le(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect()
}

/// Convert i16 samples to bytes (little-endian)
pub fn i16_to_bytes_le(samples: &[i16]) -> Vec<u8> {
    samples
        .iter()
        .flat_map(|&s| s.to_le_bytes())
        .collect()
}

/// Simple linear resampler
/// For production, use the `rubato` crate for high-quality resampling
pub struct Resampler {
    from_rate: u32,
    to_rate: u32,
}

impl Resampler {
    pub fn new(from_rate: u32, to_rate: u32) -> Self {
        Self { from_rate, to_rate }
    }

    /// Resample audio using linear interpolation
    /// For production, use rubato crate for better quality
    pub fn resample(&self, samples: &[f32]) -> Vec<f32> {
        if self.from_rate == self.to_rate {
            return samples.to_vec();
        }

        let ratio = self.to_rate as f64 / self.from_rate as f64;
        let output_len = (samples.len() as f64 * ratio).ceil() as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_idx = i as f64 / ratio;
            let src_floor = src_idx.floor() as usize;
            let src_ceil = (src_floor + 1).min(samples.len() - 1);
            let frac = (src_idx - src_floor as f64) as f32;

            let sample = if src_floor < samples.len() {
                samples[src_floor] * (1.0 - frac) + samples[src_ceil] * frac
            } else {
                0.0
            };

            output.push(sample);
        }

        debug!(
            "Resampled {} -> {} samples ({}Hz -> {}Hz)",
            samples.len(),
            output.len(),
            self.from_rate,
            self.to_rate
        );

        output
    }
}

/// Mel spectrogram extractor
/// Uses librosa-style mel filterbank
pub struct MelSpectrogram {
    /// Sample rate
    sample_rate: u32,
    /// Number of mel filterbanks
    n_mels: usize,
    /// FFT size
    n_fft: usize,
    /// Hop length in samples
    hop_length: usize,
    /// Window length in samples
    win_length: usize,
    /// Mel filterbank weights
    mel_basis: Vec<Vec<f32>>,
    /// Hann window
    window: Vec<f32>,
}

impl MelSpectrogram {
    pub fn new(
        sample_rate: u32,
        n_mels: usize,
        n_fft: usize,
        hop_length: usize,
        win_length: usize,
    ) -> Self {
        // Create Hann window
        let window: Vec<f32> = (0..win_length)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (win_length - 1) as f32).cos())
            })
            .collect();

        // Create mel filterbank (simplified)
        let mel_basis = Self::create_mel_filterbank(sample_rate, n_fft, n_mels);

        Self {
            sample_rate,
            n_mels,
            n_fft,
            hop_length,
            win_length,
            mel_basis,
            window,
        }
    }

    /// Create mel filterbank matrix
    fn create_mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
        let fmin = 0.0;
        let fmax = sample_rate as f32 / 2.0;

        // Convert Hz to mel scale
        let hz_to_mel = |hz: f32| -> f32 { 2595.0 * (1.0 + hz / 700.0).log10() };
        let mel_to_hz = |mel: f32| -> f32 { 700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0) };

        let mel_min = hz_to_mel(fmin);
        let mel_max = hz_to_mel(fmax);

        // Create mel points
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();

        // Convert back to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

        // Convert to FFT bin indices
        let fft_bins: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((n_fft + 1) as f32 * hz / sample_rate as f32).floor() as usize)
            .collect();

        // Create filterbank
        let n_freqs = n_fft / 2 + 1;
        let mut filterbank = vec![vec![0.0f32; n_freqs]; n_mels];

        for m in 0..n_mels {
            for k in fft_bins[m]..fft_bins[m + 1] {
                if k < n_freqs {
                    filterbank[m][k] = (k - fft_bins[m]) as f32
                        / (fft_bins[m + 1] - fft_bins[m]).max(1) as f32;
                }
            }
            for k in fft_bins[m + 1]..fft_bins[m + 2] {
                if k < n_freqs {
                    filterbank[m][k] = (fft_bins[m + 2] - k) as f32
                        / (fft_bins[m + 2] - fft_bins[m + 1]).max(1) as f32;
                }
            }
        }

        filterbank
    }

    /// Extract mel spectrogram from audio samples
    pub fn extract(&self, samples: &[f32]) -> Vec<Vec<f32>> {
        let mut frames = Vec::new();

        // Process each frame
        for start in (0..samples.len().saturating_sub(self.win_length)).step_by(self.hop_length) {
            let frame = &samples[start..start + self.win_length];

            // Apply window
            let windowed: Vec<f32> = frame
                .iter()
                .zip(self.window.iter())
                .map(|(&s, &w)| s * w)
                .collect();

            // Compute power spectrum (using simple DFT for now)
            // In production, use rustfft for proper FFT
            let power_spectrum = self.compute_power_spectrum(&windowed);

            // Apply mel filterbank
            let mel_features: Vec<f32> = self
                .mel_basis
                .iter()
                .map(|filter| {
                    let energy: f32 = filter
                        .iter()
                        .zip(power_spectrum.iter())
                        .map(|(&f, &p)| f * p)
                        .sum();
                    // Log mel spectrogram
                    (energy.max(1e-10)).ln()
                })
                .collect();

            frames.push(mel_features);
        }

        frames
    }

    /// Compute power spectrum using simple DFT
    /// For production, use rustfft crate
    fn compute_power_spectrum(&self, frame: &[f32]) -> Vec<f32> {
        let n = self.n_fft;
        let n_freqs = n / 2 + 1;
        let mut spectrum = vec![0.0f32; n_freqs];

        // Pad frame to n_fft
        let mut padded = vec![0.0f32; n];
        for (i, &s) in frame.iter().enumerate() {
            if i < n {
                padded[i] = s;
            }
        }

        // Simple DFT (slow but works for small frames)
        for k in 0..n_freqs {
            let mut real = 0.0f32;
            let mut imag = 0.0f32;

            for (n_idx, &x) in padded.iter().enumerate() {
                let angle = -2.0 * std::f32::consts::PI * k as f32 * n_idx as f32 / n as f32;
                real += x * angle.cos();
                imag += x * angle.sin();
            }

            spectrum[k] = (real * real + imag * imag) / n as f32;
        }

        spectrum
    }
}

impl Default for MelSpectrogram {
    fn default() -> Self {
        Self::new(
            16000,  // sample rate
            80,     // n_mels
            512,    // n_fft
            160,    // hop_length (10ms at 16kHz)
            400,    // win_length (25ms at 16kHz)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcm_conversion() {
        let i16_samples = vec![0i16, i16::MAX, i16::MIN];
        let f32_samples = pcm_i16_to_f32(&i16_samples);

        assert_eq!(f32_samples[0], 0.0);
        assert!((f32_samples[1] - 1.0).abs() < 0.001);
        assert!((f32_samples[2] + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_resampler() {
        let resampler = Resampler::new(48000, 16000);
        let input: Vec<f32> = (0..480).map(|i| (i as f32 / 100.0).sin()).collect();
        let output = resampler.resample(&input);

        // Output should be roughly 1/3 the length
        assert!(output.len() > 150 && output.len() < 170);
    }

    #[test]
    fn test_mel_spectrogram() {
        let mel = MelSpectrogram::default();
        let samples: Vec<f32> = (0..16000).map(|i| (i as f32 / 100.0).sin()).collect();
        let features = mel.extract(&samples);

        // Should have frames
        assert!(!features.is_empty());
        // Each frame should have n_mels features
        assert_eq!(features[0].len(), 80);
    }
}

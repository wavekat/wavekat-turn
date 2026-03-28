//! Pipecat Smart Turn v3 backend.
//!
//! Audio-based turn detection using the Smart Turn ONNX model.
//! Expects 16 kHz f32 PCM input. Telephony audio at 8 kHz must be
//! upsampled before feeding to this detector.
//!
//! # Model
//!
//! - Source:  <https://huggingface.co/pipecat-ai/smart-turn-v3>
//! - File:    `smart-turn-v3.2-cpu.onnx` (int8 quantized, ~8 MB)
//! - License: BSD 2-Clause
//!
//! # Tensor specification
//!
//! | Role   | Name             | Shape          | Dtype   |
//! |--------|------------------|----------------|---------|
//! | Input  | `input_features` | `[B, 80, 800]` | float32 |
//! | Output | `logits`         | `[B, 1]`       | float32 |
//!
//! Despite the name, `logits` is a **sigmoid probability** P(turn complete)
//! in [0, 1] — the sigmoid is fused into the model before ONNX export.
//! Threshold: `probability > 0.5` → `TurnState::Finished`.
//!
//! # Mel-feature specification
//!
//! The model was trained with HuggingFace `WhisperFeatureExtractor(chunk_length=8)`:
//!
//! | Parameter     | Value                          |
//! |---------------|--------------------------------|
//! | Sample rate   | 16 000 Hz                      |
//! | n_fft         | 400 samples (25 ms)            |
//! | hop_length    | 160 samples (10 ms)            |
//! | n_mels        | 80                             |
//! | Freq range    | 0 – 8 000 Hz                   |
//! | Mel scale     | Slaney (NOT HTK)               |
//! | Window        | Hann (periodic, size 400)      |
//! | Pre-emphasis  | None                           |
//! | Log           | log10 with ε = 1e-10           |
//! | Normalization | clamp(max − 8), (x + 4) / 4   |
//!
//! # Audio buffer
//!
//! - Exactly **8 seconds = 128 000 samples** at 16 kHz.
//! - Shorter input: **front-padded** with zeros (audio is at the end).
//! - Longer input: the **last** 8 s is used (oldest samples discarded).

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use ndarray::{s, Array2, Array3};
use ort::{inputs, value::Tensor};
use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::onnx;
use crate::{AudioFrame, AudioTurnDetector, StageTiming, TurnError, TurnPrediction, TurnState};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Sample rate the model expects.
const SAMPLE_RATE: u32 = 16_000;
/// FFT window size in samples (25 ms at 16 kHz).
const N_FFT: usize = 400;
/// STFT hop length in samples (10 ms at 16 kHz).
const HOP_LENGTH: usize = 160;
/// Number of mel filterbank bins.
const N_MELS: usize = 80;
/// Number of STFT frames the model expects (8 s × 100 fps).
const N_FRAMES: usize = 800;
/// FFT frequency bins: N_FFT/2 + 1.
const N_FREQS: usize = N_FFT / 2 + 1; // 201
/// Ring buffer capacity: 8 s × 16 kHz.
const RING_CAPACITY: usize = 8 * SAMPLE_RATE as usize; // 128 000

/// Embedded ONNX model bytes, downloaded by build.rs at compile time.
const MODEL_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smart-turn-v3.2-cpu.onnx"));

// ---------------------------------------------------------------------------
// Mel feature extractor
// ---------------------------------------------------------------------------

/// Pre-computed Whisper-style log-mel feature extractor.
///
/// All expensive setup (filterbank, window, FFT plan) happens once in [`new`].
/// [`MelExtractor::extract`] is then called per inference.
struct MelExtractor {
    /// Slaney-normalised mel filterbank: shape [N_MELS, N_FREQS].
    mel_filters: Array2<f32>,
    /// Periodic Hann window of length N_FFT.
    hann_window: Vec<f32>,
    /// Reusable forward real FFT plan.
    fft: Arc<dyn RealToComplex<f32>>,
    /// Reusable scratch buffer for the FFT.
    fft_scratch: Vec<Complex<f32>>,
    /// Reusable output spectrum buffer (N_FREQS complex values).
    spectrum_buf: Vec<Complex<f32>>,
    /// Cached power spectrogram [N_FREQS × (N_FRAMES+1)] from the previous call.
    /// Enables incremental STFT: only new frames are recomputed.
    cached_power_spec: Option<Array2<f32>>,
    /// Cached mel spectrogram [N_MELS × N_FRAMES] from the previous call.
    /// Enables incremental mel filterbank: only new columns are recomputed.
    cached_mel_spec: Option<Array2<f32>>,
}

impl MelExtractor {
    fn new() -> Self {
        let mel_filters = build_mel_filters(
            SAMPLE_RATE as usize,
            N_FFT,
            N_MELS,
            0.0,
            SAMPLE_RATE as f32 / 2.0,
        );
        let hann_window = periodic_hann(N_FFT);

        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let fft_scratch = fft.make_scratch_vec();
        let spectrum_buf = fft.make_output_vec();

        Self {
            mel_filters,
            hann_window,
            fft,
            fft_scratch,
            spectrum_buf,
            cached_power_spec: None,
            cached_mel_spec: None,
        }
    }

    /// Compute a [N_MELS × N_FRAMES] log-mel spectrogram from exactly
    /// `RING_CAPACITY` samples of 16 kHz mono audio.
    ///
    /// `shift_frames` is how many STFT frames worth of new audio were added
    /// since the last call. When a valid cache exists and `shift_frames` is
    /// in range, only the last `shift_frames` columns of the power spectrogram
    /// are recomputed; the rest are copied from the shifted cache.
    fn extract(&mut self, audio: &[f32], shift_frames: usize) -> Array2<f32> {
        debug_assert_eq!(audio.len(), RING_CAPACITY);

        // ---- Center-pad: N_FFT/2 zeros on each side → 128 400 samples ----
        // This replicates librosa/PyTorch `center=True` STFT behaviour, which
        // gives exactly N_FRAMES + 1 = 801 frames; we discard the last one.
        let pad = N_FFT / 2;
        let mut padded = vec![0.0f32; pad + audio.len() + pad];
        padded[pad..pad + audio.len()].copy_from_slice(audio);

        // n_total = (128 400 − 400) / 160 + 1 = 801
        let n_total_frames = (padded.len() - N_FFT) / HOP_LENGTH + 1;

        // ---- Incremental STFT ----
        // If we have a cached power spec and shift_frames < n_total_frames,
        // reuse the unchanged frames by shifting the cache left and only
        // computing the `shift_frames` new columns at the end.
        let first_new_frame = match &self.cached_power_spec {
            Some(cached) if shift_frames > 0 && shift_frames < n_total_frames => {
                let kept = n_total_frames - shift_frames;
                let mut power_spec = Array2::<f32>::zeros((N_FREQS, n_total_frames));
                power_spec
                    .slice_mut(s![.., ..kept])
                    .assign(&cached.slice(s![.., shift_frames..]));
                self.cached_power_spec = Some(power_spec);
                kept // only compute frames [kept..n_total_frames]
            }
            _ => {
                self.cached_power_spec = Some(Array2::<f32>::zeros((N_FREQS, n_total_frames)));
                0 // cold start: compute all frames
            }
        };

        let power_spec = self.cached_power_spec.as_mut().unwrap();
        let mut frame_buf = vec![0.0f32; N_FFT];

        for frame_idx in first_new_frame..n_total_frames {
            let start = frame_idx * HOP_LENGTH;
            // Apply periodic Hann window
            for (i, (&s, &w)) in padded[start..start + N_FFT]
                .iter()
                .zip(self.hann_window.iter())
                .enumerate()
            {
                frame_buf[i] = s * w;
            }

            self.fft
                .process_with_scratch(
                    &mut frame_buf,
                    &mut self.spectrum_buf,
                    &mut self.fft_scratch,
                )
                .expect("FFT failed: internal buffer size mismatch");

            for (k, c) in self.spectrum_buf.iter().enumerate() {
                power_spec[[k, frame_idx]] = c.re * c.re + c.im * c.im;
            }
        }

        // Take first N_FRAMES columns (drop the trailing frame)
        let power_spec_view = power_spec.slice(s![.., ..N_FRAMES]);

        // ---- Incremental mel filterbank: [N_MELS, N_FREQS] × [N_FREQS, shift_frames] ----
        // Reuse the cached mel columns for the unchanged frames; only multiply
        // the new power-spectrum columns against the filterbank.
        let mel_spec = match &self.cached_mel_spec {
            Some(cached) if shift_frames > 0 && shift_frames <= N_FRAMES => {
                let kept = N_FRAMES - shift_frames;
                let mut ms = Array2::<f32>::zeros((N_MELS, N_FRAMES));
                // Shift old columns left
                ms.slice_mut(s![.., ..kept])
                    .assign(&cached.slice(s![.., shift_frames..]));
                // Apply filterbank only to the new power-spectrum columns
                let new_power = power_spec_view.slice(s![.., kept..]);
                ms.slice_mut(s![.., kept..])
                    .assign(&self.mel_filters.dot(&new_power));
                ms
            }
            _ => self.mel_filters.dot(&power_spec_view),
        };
        self.cached_mel_spec = Some(mel_spec.clone());

        // ---- Log10 with floor at 1e-10 ----
        let mut log_mel = mel_spec.mapv(|x| x.max(1e-10_f32).log10());

        // ---- Dynamic range compression and normalization ----
        // Matches WhisperFeatureExtractor: clamp to [max−8, ∞], then (x+4)/4
        let max_val = log_mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        log_mel.mapv_inplace(|x| (x.max(max_val - 8.0) + 4.0) / 4.0);

        log_mel
    }

    /// Invalidate all caches (call on reset).
    fn invalidate_cache(&mut self) {
        self.cached_power_spec = None;
        self.cached_mel_spec = None;
    }
}

// ---------------------------------------------------------------------------
// Mel filterbank construction — Slaney scale, slaney norm
// ---------------------------------------------------------------------------

/// Convert Hz to mel (Slaney/librosa scale, NOT HTK).
fn hz_to_mel(hz: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0; // linear region slope (Hz per mel)
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP; // = 15.0
                                                // logstep = ln(6.4) / 27  (≈ 0.068752)
    let logstep = (6.4_f32).ln() / 27.0;
    if hz >= MIN_LOG_HZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HZ).ln() / logstep
    } else {
        hz / F_SP
    }
}

/// Convert mel back to Hz (Slaney scale).
fn mel_to_hz(mel: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    const MIN_LOG_MEL: f32 = MIN_LOG_HZ / F_SP;
    let logstep = (6.4_f32).ln() / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HZ * ((mel - MIN_LOG_MEL) * logstep).exp()
    } else {
        mel * F_SP
    }
}

/// Build a Slaney-normalised mel filterbank of shape [n_mels, n_freqs].
///
/// Matches `librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax,
///   norm="slaney", dtype=float32)` which is what HuggingFace's
/// `WhisperFeatureExtractor` uses internally.
fn build_mel_filters(
    sr: usize,
    n_fft: usize,
    n_mels: usize,
    f_min: f32,
    f_max: f32,
) -> Array2<f32> {
    let n_freqs = n_fft / 2 + 1;

    // FFT frequency bins: 0, sr/n_fft, 2·sr/n_fft, …
    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sr as f32 / n_fft as f32)
        .collect();

    // n_mels + 2 equally-spaced mel points (edge + n_mels centres + edge)
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);
    let mel_pts: Vec<f32> = (0..=(n_mels + 1))
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
        .collect();
    let hz_pts: Vec<f32> = mel_pts.iter().map(|&m| mel_to_hz(m)).collect();

    // Build triangular filters with Slaney normalisation
    let mut filters = Array2::<f32>::zeros((n_mels, n_freqs));
    for m in 0..n_mels {
        let f_left = hz_pts[m];
        let f_center = hz_pts[m + 1];
        let f_right = hz_pts[m + 2];
        // Slaney norm: 2 / (right_hz − left_hz)
        let enorm = 2.0 / (f_right - f_left);

        for (k, &f) in fft_freqs.iter().enumerate() {
            let w = if f >= f_left && f <= f_center {
                (f - f_left) / (f_center - f_left)
            } else if f > f_center && f <= f_right {
                (f_right - f) / (f_right - f_center)
            } else {
                0.0
            };
            filters[[m, k]] = w * enorm;
        }
    }
    filters
}

// ---------------------------------------------------------------------------
// Hann window
// ---------------------------------------------------------------------------

/// Periodic Hann window of length `n`, matching `torch.hann_window(n, periodic=True)`.
///
/// Formula: `w[k] = 0.5 · (1 − cos(2π·k / n))` for k in 0..n.
/// This differs from the symmetric variant (which divides by n−1).
fn periodic_hann(n: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    (0..n)
        .map(|k| 0.5 * (1.0 - (2.0 * PI * k as f32 / n as f32).cos()))
        .collect()
}

// ---------------------------------------------------------------------------
// Audio preparation
// ---------------------------------------------------------------------------

/// Pad or truncate `samples` to exactly `RING_CAPACITY` samples.
///
/// - Longer: keep the **last** 8 s (discard oldest).
/// - Shorter: **front-pad** with zeros so audio is right-aligned.
fn prepare_audio(samples: &[f32]) -> Vec<f32> {
    match samples.len().cmp(&RING_CAPACITY) {
        std::cmp::Ordering::Equal => samples.to_vec(),
        std::cmp::Ordering::Greater => samples[samples.len() - RING_CAPACITY..].to_vec(),
        std::cmp::Ordering::Less => {
            let mut out = vec![0.0f32; RING_CAPACITY - samples.len()];
            out.extend_from_slice(samples);
            out
        }
    }
}

// ---------------------------------------------------------------------------
// PipecatSmartTurn
// ---------------------------------------------------------------------------

/// Pipecat Smart Turn v3 detector.
///
/// Buffers up to 8 seconds of audio internally. Call [`push_audio`] with
/// every incoming 16 kHz frame, then call [`predict`] when the VAD fires
/// end-of-speech to get a [`TurnPrediction`].
///
/// # Usage with VAD
///
/// ```no_run
/// # #[cfg(feature = "pipecat")]
/// # {
/// use wavekat_turn::audio::PipecatSmartTurn;
/// use wavekat_turn::AudioTurnDetector;
///
/// let mut detector = PipecatSmartTurn::new().unwrap();
/// // ... feed frames via push_audio ...
/// let prediction = detector.predict().unwrap();
/// println!("{:?} ({:.2})", prediction.state, prediction.confidence);
/// # }
/// ```
///
/// [`push_audio`]: AudioTurnDetector::push_audio
/// [`predict`]: AudioTurnDetector::predict
pub struct PipecatSmartTurn {
    session: ort::session::Session,
    ring_buffer: VecDeque<f32>,
    mel: MelExtractor,
    /// Counts samples pushed since the last `predict()` call.
    /// Used to compute `shift_frames` for incremental STFT.
    samples_since_predict: usize,
}

// SAFETY: ort::Session is Send in ort 2.x. Sync is safe because every
// method that touches the session takes &mut self, preventing concurrent use.
unsafe impl Send for PipecatSmartTurn {}
unsafe impl Sync for PipecatSmartTurn {}

impl PipecatSmartTurn {
    /// Load the Smart Turn v3.2 model embedded at compile time.
    pub fn new() -> Result<Self, TurnError> {
        let session = onnx::session_from_memory(MODEL_BYTES)?;
        Ok(Self::build(session))
    }

    /// Load a model from a custom path on disk.
    ///
    /// Useful for CI environments that supply the model file separately, or
    /// for evaluating fine-tuned variants without recompiling.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TurnError> {
        let session = onnx::session_from_file(path)?;
        Ok(Self::build(session))
    }

    fn build(session: ort::session::Session) -> Self {
        Self {
            session,
            ring_buffer: VecDeque::with_capacity(RING_CAPACITY),
            mel: MelExtractor::new(),
            samples_since_predict: 0,
        }
    }
}

impl AudioTurnDetector for PipecatSmartTurn {
    /// Append audio to the internal ring buffer.
    ///
    /// Frames with a sample rate other than 16 kHz are silently dropped.
    /// The ring buffer holds at most 8 s; older samples are evicted.
    fn push_audio(&mut self, frame: &AudioFrame) {
        if frame.sample_rate() != SAMPLE_RATE {
            return;
        }
        let samples = frame.samples();
        // Evict oldest samples to make room
        let overflow = (self.ring_buffer.len() + samples.len()).saturating_sub(RING_CAPACITY);
        if overflow > 0 {
            self.ring_buffer.drain(..overflow);
        }
        self.ring_buffer.extend(samples.iter().copied());
        self.samples_since_predict += samples.len();
    }

    /// Run inference on the buffered audio.
    ///
    /// Takes a snapshot of the ring buffer, pads/truncates to 8 s, extracts
    /// Whisper log-mel features, and runs ONNX inference.
    fn predict(&mut self) -> Result<TurnPrediction, TurnError> {
        let t_start = Instant::now();

        // Stage 1: Snapshot the ring buffer and prepare exactly 128 000 samples
        let shift_frames = self.samples_since_predict / HOP_LENGTH;
        self.samples_since_predict = 0;

        let buffered: Vec<f32> = self.ring_buffer.iter().copied().collect();
        let audio = prepare_audio(&buffered);
        let t_after_audio_prep = Instant::now();

        // Stage 2: Extract [N_MELS × N_FRAMES] log-mel features (incremental)
        let mel_spec = self.mel.extract(&audio, shift_frames);
        let t_after_mel = Instant::now();

        // Stage 3: Reshape to [1, N_MELS, N_FRAMES] and run ONNX inference
        let (raw, _) = mel_spec.into_raw_vec_and_offset();
        let input_array = Array3::from_shape_vec((1, N_MELS, N_FRAMES), raw)
            .expect("internal: mel output has wrong element count");

        let input_tensor = Tensor::from_array(input_array)
            .map_err(|e| TurnError::BackendError(format!("failed to create input tensor: {e}")))?;

        let outputs = self
            .session
            .run(inputs!["input_features" => input_tensor])
            .map_err(|e| TurnError::BackendError(format!("inference failed: {e}")))?;
        let t_after_onnx = Instant::now();

        // Extract sigmoid probability from the "logits" output
        let output = outputs
            .get("logits")
            .ok_or_else(|| TurnError::BackendError("missing 'logits' output tensor".into()))?;
        let (_, data): (_, &[f32]) = output
            .try_extract_tensor()
            .map_err(|e| TurnError::BackendError(format!("failed to extract logits: {e}")))?;
        let probability = *data
            .first()
            .ok_or_else(|| TurnError::BackendError("logits tensor is empty".into()))?;

        let latency_ms = t_start.elapsed().as_millis() as u64;

        let us = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1_000_000.0;
        let stage_times = vec![
            StageTiming {
                name: "audio_prep",
                us: us(t_start, t_after_audio_prep),
            },
            StageTiming {
                name: "mel",
                us: us(t_after_audio_prep, t_after_mel),
            },
            StageTiming {
                name: "onnx",
                us: us(t_after_mel, t_after_onnx),
            },
        ];

        // probability = P(turn complete); > 0.5 means the speaker has finished
        let (state, confidence) = if probability > 0.5 {
            (TurnState::Finished, probability)
        } else {
            (TurnState::Unfinished, 1.0 - probability)
        };

        Ok(TurnPrediction {
            state,
            confidence,
            latency_ms,
            stage_times,
        })
    }

    /// Clear the ring buffer. Call at the start of each new speech turn.
    fn reset(&mut self) {
        self.ring_buffer.clear();
        self.samples_since_predict = 0;
        self.mel.invalidate_cache();
    }
}

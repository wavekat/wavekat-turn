//! # wavekat-turn
//!
//! Unified turn detection with multiple backends.
//!
//! Provides a clean abstraction over turn-detection models that predict
//! whether a user has finished speaking. Two trait families cover the
//! two fundamental input modalities:
//!
//! - [`AudioTurnDetector`] — operates on raw audio frames (e.g. Pipecat Smart Turn)
//! - [`TextTurnDetector`] — operates on ASR transcript text (e.g. LiveKit EOU)
//!
//! For most use cases, wrap a detector in [`TurnController`] to get
//! automatic state tracking and soft-reset logic for VAD integration.
//! See [`controller`] for details.
//!
//! # Feature flags
//!
//! | Feature | Backend | Input |
//! |---------|---------|-------|
//! | `pipecat` | Pipecat Smart Turn v3 (ONNX) | Audio (16 kHz) |
//! | `livekit` | LiveKit Turn Detector (ONNX) | Text |

pub mod controller;
pub mod error;

#[cfg(any(feature = "pipecat", feature = "livekit"))]
pub(crate) mod onnx;

#[cfg(feature = "pipecat")]
pub mod audio;

#[cfg(feature = "livekit")]
pub mod text;

pub use controller::TurnController;
pub use error::TurnError;
pub use wavekat_core::AudioFrame;

/// The predicted turn state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnState {
    /// User is done speaking — AI should respond.
    Finished,
    /// User is still speaking or thinking.
    Unfinished,
    /// User explicitly asked the AI to wait.
    Wait,
}

/// Per-stage timing entry.
#[derive(Debug, Clone)]
pub struct StageTiming {
    /// Stage name (e.g. "audio_prep", "mel", "onnx").
    pub name: &'static str,
    /// Time in microseconds for this stage.
    pub us: f64,
}

/// A turn detection prediction with confidence and timing metadata.
#[derive(Debug, Clone)]
pub struct TurnPrediction {
    pub state: TurnState,
    pub confidence: f32,
    pub latency_ms: u64,
    /// Per-stage timing breakdown in pipeline order.
    pub stage_times: Vec<StageTiming>,
    /// Duration of audio in the detector's buffer at prediction time (ms).
    ///
    /// For PipecatSmartTurn this reflects how much of the 8 s ring buffer
    /// was filled. With soft reset the buffer may span multiple speech
    /// segments, so this can exceed the current segment duration.
    pub audio_duration_ms: u64,
}

/// A single turn in the conversation, for context-aware text detectors.
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub role: Role,
    pub text: String,
}

/// Speaker role in a conversation turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

/// Turn detector that operates on raw audio.
///
/// Implementations buffer audio internally and run prediction on demand.
///
/// **Most users should wrap this in [`TurnController`]** rather than calling
/// these methods directly. The controller tracks prediction state and provides
/// [`reset_if_finished`](TurnController::reset_if_finished) for correct
/// multi-utterance handling.
///
/// # Direct usage (advanced)
///
/// If you need full control over reset logic:
///
/// 1. **Every audio chunk** → [`push_audio`](AudioTurnDetector::push_audio)
/// 2. **VAD fires "speech stopped"** → [`predict`](AudioTurnDetector::predict)
/// 3. **New turn begins** → [`reset`](AudioTurnDetector::reset)
///
/// Note: calling `reset` unconditionally on every VAD speech-start will discard
/// audio context when the user pauses mid-sentence. See [`TurnController`] for
/// the recommended approach.
pub trait AudioTurnDetector: Send + Sync {
    /// Feed audio into the internal buffer.
    ///
    /// Call continuously with incoming audio frames (16 kHz mono).
    fn push_audio(&mut self, frame: &AudioFrame);

    /// Run prediction on buffered audio.
    ///
    /// Call when VAD detects end of speech. The buffer is **not** cleared
    /// after prediction — call [`reset`](AudioTurnDetector::reset) explicitly
    /// when starting a new turn.
    fn predict(&mut self) -> Result<TurnPrediction, TurnError>;

    /// Unconditionally clear the internal buffer.
    ///
    /// Use when you are certain a new turn is starting (e.g. after the
    /// assistant finishes responding). For VAD speech-start events where
    /// the user may be continuing, prefer
    /// [`TurnController::reset_if_finished`].
    fn reset(&mut self);
}

/// Turn detector that operates on ASR transcript text.
///
/// Implementations receive the current (possibly partial) transcript
/// and optionally prior conversation turns for context.
pub trait TextTurnDetector: Send + Sync {
    fn predict_text(
        &mut self,
        transcript: &str,
        context: &[ConversationTurn],
    ) -> Result<TurnPrediction, TurnError>;
    fn reset(&mut self);
}

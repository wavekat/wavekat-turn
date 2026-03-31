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
/// The typical flow with VAD:
///
/// 1. **Every audio chunk** → [`push_audio`](AudioTurnDetector::push_audio)
/// 2. **VAD fires "speech started"** → [`reset`](AudioTurnDetector::reset)
/// 3. **VAD fires "speech stopped"** → [`predict`](AudioTurnDetector::predict)
pub trait AudioTurnDetector: Send + Sync {
    /// Feed audio into the internal buffer.
    ///
    /// Call continuously with incoming audio frames (16 kHz mono).
    fn push_audio(&mut self, frame: &AudioFrame);

    /// Run prediction on buffered audio.
    ///
    /// Call when VAD detects end of speech.
    fn predict(&mut self) -> Result<TurnPrediction, TurnError>;

    /// Clear the internal buffer. Call when a new speech turn begins.
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

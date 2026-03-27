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

pub mod error;

#[cfg(feature = "pipecat")]
pub mod audio;

#[cfg(feature = "livekit")]
pub mod text;

pub use error::TurnError;

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

/// A turn detection prediction with confidence and timing metadata.
#[derive(Debug, Clone)]
pub struct TurnPrediction {
    pub state: TurnState,
    pub confidence: f32,
    pub latency_ms: u64,
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

/// Turn detector that operates on raw audio frames.
///
/// Implementations receive 16 kHz f32 PCM and return a turn prediction.
pub trait AudioTurnDetector: Send + Sync {
    fn predict_audio(&mut self, audio: &[f32]) -> Result<TurnPrediction, TurnError>;
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

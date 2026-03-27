//! LiveKit Turn Detector backend.
//!
//! Text-based end-of-utterance detection using LiveKit's distilled
//! Qwen2.5-0.5B ONNX model. Operates on ASR transcript text with
//! optional conversation context.
//!
//! - Model size: ~400 MB
//! - Inference: ~25 ms on CPU
//! - License: LiveKit Model License

use crate::{ConversationTurn, TextTurnDetector, TurnError, TurnPrediction};

/// LiveKit end-of-utterance detector.
pub struct LiveKitEou {
    // TODO: ONNX session + tokenizer
}

impl LiveKitEou {
    /// Create a new LiveKit turn detector, loading the ONNX model.
    pub fn new() -> Result<Self, TurnError> {
        todo!("load LiveKit EOU ONNX model")
    }
}

impl TextTurnDetector for LiveKitEou {
    fn predict_text(
        &mut self,
        _transcript: &str,
        _context: &[ConversationTurn],
    ) -> Result<TurnPrediction, TurnError> {
        todo!("run ONNX inference")
    }

    fn reset(&mut self) {
        todo!("reset internal state")
    }
}

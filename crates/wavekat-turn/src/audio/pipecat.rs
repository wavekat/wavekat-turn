//! Pipecat Smart Turn v3 backend.
//!
//! Audio-based turn detection using the Smart Turn ONNX model.
//! Expects 16 kHz f32 PCM input. Telephony audio at 8 kHz must be
//! upsampled before feeding to this detector.
//!
//! - Model size: ~8 MB (int8 quantized ONNX)
//! - Inference: ~12 ms on CPU
//! - License: BSD 2-Clause

use crate::{AudioFrame, AudioTurnDetector, TurnError, TurnPrediction};

/// Pipecat Smart Turn v3 detector.
///
/// Buffers up to 8 seconds of audio internally. When [`predict`](AudioTurnDetector::predict)
/// is called, it takes the last 8s (zero-padded at front if shorter),
/// extracts Whisper log-mel features, and runs ONNX inference.
pub struct PipecatSmartTurn {
    // TODO: ONNX session + audio ring buffer + state
}

impl PipecatSmartTurn {
    /// Create a new Smart Turn detector, loading the ONNX model.
    pub fn new() -> Result<Self, TurnError> {
        todo!("load Smart Turn v3 ONNX model")
    }
}

impl AudioTurnDetector for PipecatSmartTurn {
    fn push_audio(&mut self, _frame: &AudioFrame) {
        todo!("append to ring buffer")
    }

    fn predict(&mut self) -> Result<TurnPrediction, TurnError> {
        todo!("truncate/pad to 8s, extract mel features, run ONNX inference")
    }

    fn reset(&mut self) {
        todo!("clear ring buffer and internal state")
    }
}

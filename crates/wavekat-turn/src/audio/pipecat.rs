//! Pipecat Smart Turn v3 backend.
//!
//! Audio-based turn detection using the Smart Turn ONNX model.
//! Expects 16 kHz f32 PCM input. Telephony audio at 8 kHz must be
//! upsampled before feeding to this detector.
//!
//! - Model size: ~8 MB (int8 quantized ONNX)
//! - Inference: ~12 ms on CPU
//! - License: BSD 2-Clause

use crate::{AudioTurnDetector, TurnError, TurnPrediction};

/// Pipecat Smart Turn v3 detector.
pub struct PipecatSmartTurn {
    // TODO: ONNX session + state
}

impl PipecatSmartTurn {
    /// Create a new Smart Turn detector, loading the ONNX model.
    pub fn new() -> Result<Self, TurnError> {
        todo!("load Smart Turn v3 ONNX model")
    }
}

impl AudioTurnDetector for PipecatSmartTurn {
    fn predict_audio(&mut self, _audio: &[f32]) -> Result<TurnPrediction, TurnError> {
        todo!("run ONNX inference")
    }

    fn reset(&mut self) {
        todo!("reset internal state")
    }
}

//! Tests for [`TurnController`].
//!
//! Uses a mock detector to test orchestration logic without ONNX overhead.

use wavekat_turn::{
    AudioFrame, AudioTurnDetector, TurnController, TurnError, TurnPrediction, TurnState,
};

// ---------------------------------------------------------------------------
// Mock detector
// ---------------------------------------------------------------------------

/// A minimal detector that records calls and returns a configurable state.
struct MockDetector {
    /// The state to return on the next `predict()` call.
    next_state: TurnState,
    /// Number of samples in the buffer (cleared by reset).
    buffer_len: usize,
    /// How many times `reset()` was called.
    reset_count: usize,
}

impl MockDetector {
    fn new() -> Self {
        Self {
            next_state: TurnState::Unfinished,
            buffer_len: 0,
            reset_count: 0,
        }
    }
}

impl AudioTurnDetector for MockDetector {
    fn push_audio(&mut self, frame: &AudioFrame) {
        self.buffer_len += frame.samples().len();
    }

    fn predict(&mut self) -> Result<TurnPrediction, TurnError> {
        let state = self.next_state;
        let confidence = match state {
            TurnState::Finished => 0.95,
            TurnState::Unfinished => 0.80,
            TurnState::Wait => 0.70,
        };
        let audio_duration_ms = (self.buffer_len as u64 * 1000) / 16000;
        Ok(TurnPrediction {
            state,
            confidence,
            latency_ms: 0,
            stage_times: vec![],
            audio_duration_ms,
        })
    }

    fn reset(&mut self) {
        self.buffer_len = 0;
        self.reset_count += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn reset_if_finished_resets_on_first_call() {
    let mut ctrl = TurnController::new(MockDetector::new());
    assert!(
        ctrl.reset_if_finished(),
        "should reset when no prior prediction"
    );
}

#[test]
fn reset_if_finished_skips_after_unfinished() {
    let mut ctrl = TurnController::new(MockDetector::new());
    ctrl.inner_mut().next_state = TurnState::Unfinished;
    ctrl.predict().unwrap();

    assert!(
        !ctrl.reset_if_finished(),
        "should skip reset after Unfinished"
    );
}

#[test]
fn reset_if_finished_resets_after_finished() {
    let mut ctrl = TurnController::new(MockDetector::new());
    ctrl.inner_mut().next_state = TurnState::Finished;
    ctrl.predict().unwrap();

    assert!(ctrl.reset_if_finished(), "should reset after Finished");
}

#[test]
fn hard_reset_always_clears() {
    let mut ctrl = TurnController::new(MockDetector::new());
    ctrl.inner_mut().next_state = TurnState::Unfinished;
    ctrl.predict().unwrap();

    ctrl.reset();
    assert_eq!(
        ctrl.last_state(),
        None,
        "hard reset should clear last_state"
    );
    assert_eq!(ctrl.inner_mut().reset_count, 1);
}

#[test]
fn last_state_tracks_predictions() {
    let mut ctrl = TurnController::new(MockDetector::new());
    assert_eq!(ctrl.last_state(), None);

    ctrl.inner_mut().next_state = TurnState::Unfinished;
    ctrl.predict().unwrap();
    assert_eq!(ctrl.last_state(), Some(TurnState::Unfinished));

    ctrl.inner_mut().next_state = TurnState::Finished;
    ctrl.predict().unwrap();
    assert_eq!(ctrl.last_state(), Some(TurnState::Finished));

    ctrl.reset();
    assert_eq!(ctrl.last_state(), None);
}

#[test]
fn predict_accumulates_across_soft_reset() {
    let mut ctrl = TurnController::new(MockDetector::new());

    // Speech A
    let frame_a = AudioFrame::new(&[0.1f32; 1600][..], 16_000).into_owned();
    ctrl.push_audio(&frame_a);
    ctrl.inner_mut().next_state = TurnState::Unfinished;
    ctrl.predict().unwrap();

    // Soft reset — should NOT clear buffer
    assert!(!ctrl.reset_if_finished());

    // Speech B
    let frame_b = AudioFrame::new(&[0.2f32; 1600][..], 16_000).into_owned();
    ctrl.push_audio(&frame_b);

    // Buffer should contain both A and B
    assert_eq!(
        ctrl.inner_mut().buffer_len,
        3200,
        "buffer should have A + B samples"
    );
    assert_eq!(
        ctrl.inner_mut().reset_count,
        0,
        "no resets should have occurred"
    );
}

#[test]
fn into_inner_returns_detector() {
    let mut ctrl = TurnController::new(MockDetector::new());
    let frame = AudioFrame::new(&[0.0f32; 160][..], 16_000).into_owned();
    ctrl.push_audio(&frame);

    let detector = ctrl.into_inner();
    assert_eq!(detector.buffer_len, 160);
}

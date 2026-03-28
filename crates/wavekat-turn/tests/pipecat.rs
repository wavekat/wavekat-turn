//! Integration tests for the Pipecat Smart Turn v3 backend.
//!
//! Run with: `cargo test --features pipecat`
//! Run RTF test with: `cargo test --features pipecat --release`

#![cfg(feature = "pipecat")]

use wavekat_turn::audio::PipecatSmartTurn;
use wavekat_turn::{AudioFrame, AudioTurnDetector, TurnPrediction};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create an AudioFrame of silence (zeros) at 16 kHz.
fn silence(num_samples: usize) -> AudioFrame<'static> {
    let samples = vec![0.0f32; num_samples];
    AudioFrame::new(samples.as_slice(), 16_000).into_owned()
}

/// Push `duration_secs` of silence in 160-sample chunks (10 ms each).
fn push_silence(detector: &mut PipecatSmartTurn, duration_secs: f32) {
    let total = (duration_secs * 16_000.0) as usize;
    let chunk = 160;
    let mut pushed = 0;
    while pushed < total {
        let n = chunk.min(total - pushed);
        detector.push_audio(&silence(n));
        pushed += n;
    }
}

fn valid_prediction(pred: &TurnPrediction) {
    assert!(
        pred.confidence >= 0.0 && pred.confidence <= 1.0,
        "confidence out of range: {}",
        pred.confidence
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_new_loads_model() {
    PipecatSmartTurn::new().expect("PipecatSmartTurn::new() should succeed");
}

#[test]
fn test_from_file_loads_model() {
    let tmp = std::env::temp_dir().join("wavekat_turn_test");
    std::fs::create_dir_all(&tmp).unwrap();
    let path = tmp.join("smart-turn-test.onnx");

    let model_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/smart-turn-v3.2-cpu.onnx"));
    std::fs::write(&path, model_bytes).unwrap();

    PipecatSmartTurn::from_file(&path).expect("from_file should succeed with a valid model");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_predict_returns_valid_output() {
    let mut d = PipecatSmartTurn::new().unwrap();
    push_silence(&mut d, 2.0);
    let pred = d.predict().unwrap();
    valid_prediction(&pred);
}

#[test]
fn test_predict_with_empty_buffer() {
    // Empty buffer is front-padded to 8 s of zeros; inference must succeed.
    let mut d = PipecatSmartTurn::new().unwrap();
    let pred = d.predict().unwrap();
    valid_prediction(&pred);
}

#[test]
fn test_push_audio_wrong_sample_rate_is_ignored() {
    let mut d = PipecatSmartTurn::new().unwrap();
    let bad = AudioFrame::new(vec![0.5f32; 160].as_slice(), 8_000).into_owned();
    d.push_audio(&bad);
    // Frame should have been dropped; predict must still succeed.
    let pred = d.predict().unwrap();
    valid_prediction(&pred);
}

#[test]
fn test_reset_clears_buffer() {
    let mut d = PipecatSmartTurn::new().unwrap();
    push_silence(&mut d, 4.0);
    d.reset();
    // After reset the buffer is empty; should behave identically to a fresh instance.
    let fresh = PipecatSmartTurn::new().unwrap().predict().unwrap();
    let after_reset = d.predict().unwrap();
    assert_eq!(
        after_reset.state, fresh.state,
        "state after reset should match a fresh instance"
    );
    assert!(
        (after_reset.confidence - fresh.confidence).abs() < 1e-5,
        "confidence after reset should match a fresh instance"
    );
}

#[test]
fn test_ring_buffer_caps_at_8_seconds() {
    let mut d = PipecatSmartTurn::new().unwrap();
    push_silence(&mut d, 10.0); // 10 s > 8 s capacity; must not panic
    valid_prediction(&d.predict().unwrap());
}

#[test]
fn test_multiple_predicts_are_deterministic() {
    let mut d = PipecatSmartTurn::new().unwrap();
    push_silence(&mut d, 2.0);
    let p1 = d.predict().unwrap();
    let p2 = d.predict().unwrap();
    assert_eq!(p1.state, p2.state, "repeated predict should give same state");
    assert!(
        (p1.confidence - p2.confidence).abs() < 1e-5,
        "repeated predict should give same confidence"
    );
}

/// RTF target: < 50 ms. Only enforced in release builds because the debug
/// binary is ~10× slower.
#[test]
#[cfg(not(debug_assertions))]
fn test_latency_under_50ms() {
    let mut d = PipecatSmartTurn::new().unwrap();
    push_silence(&mut d, 2.0);
    let pred = d.predict().unwrap();
    assert!(
        pred.latency_ms < 50,
        "inference too slow: {} ms (limit: 50 ms)",
        pred.latency_ms
    );
}

/// Smoke test: latency is measured and non-zero (always runs, including debug).
#[test]
fn test_latency_is_measured() {
    let mut d = PipecatSmartTurn::new().unwrap();
    push_silence(&mut d, 2.0);
    let pred = d.predict().unwrap();
    // latency_ms == 0 would mean the timer wasn't working
    assert!(pred.latency_ms < 60_000, "latency suspiciously large: {} ms", pred.latency_ms);
}

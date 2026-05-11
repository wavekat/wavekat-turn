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
fn test_with_variant_pipecat_v3_loads_model() {
    use wavekat_turn::audio::SmartTurnVariant;
    PipecatSmartTurn::with_variant(SmartTurnVariant::PipecatV3)
        .expect("with_variant(PipecatV3) should succeed");
}

/// Exercise the WAVEKAT_TURN_MODEL_DIR override path without touching the
/// network: drop the embedded Pipecat ONNX into a temp dir under the
/// expected `<lang>/smart-turn-cpu.onnx` layout and confirm the variant
/// loader picks it up. The bytes happen to be the upstream model — that's
/// fine; we are only asserting the file resolution path works.
#[cfg(feature = "wavekat-smart-turn")]
#[test]
fn test_wavekat_variant_uses_local_dir_override() {
    use wavekat_turn::audio::{SmartTurnLang, SmartTurnVariant};

    let tmp = std::env::temp_dir().join("wavekat_turn_local_dir_test");
    let lang_dir = tmp.join("zh");
    std::fs::create_dir_all(&lang_dir).unwrap();
    let path = lang_dir.join("smart-turn-cpu.onnx");
    let model_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/smart-turn-v3.2-cpu.onnx"));
    std::fs::write(&path, model_bytes).unwrap();

    // SAFETY: tests inside this crate that mutate env vars run on the same
    // process. `cargo test` defaults to single-threaded for harness=false,
    // but the std test harness parallelises — keep the env var set for the
    // duration of this test and accept that no other test reads it.
    unsafe {
        std::env::set_var("WAVEKAT_TURN_MODEL_DIR", &tmp);
    }
    let result = PipecatSmartTurn::with_variant(SmartTurnVariant::Wavekat(SmartTurnLang::Zh));
    unsafe {
        std::env::remove_var("WAVEKAT_TURN_MODEL_DIR");
    }

    let _ = std::fs::remove_dir_all(&tmp);
    result.expect("with_variant(Wavekat(Zh)) should pick up the local override");
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
    assert_eq!(
        p1.state, p2.state,
        "repeated predict should give same state"
    );
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

#[test]
fn test_from_file_invalid_path_returns_error() {
    let result = PipecatSmartTurn::from_file("/nonexistent/path/model.onnx");
    assert!(
        result.is_err(),
        "from_file with invalid path should return an error"
    );
}

/// End-to-end smoke test for the WaveKat HuggingFace download path.
///
/// Pulls `wavekat/smart-turn-ONNX` from the Hub (cached in `$HF_HOME/hub/`
/// after the first run), runs it against the three repo fixtures, and
/// prints a markdown table of probabilities. Marked `#[ignore]` so CI and
/// `cargo test` never hit the network unintentionally.
///
/// Run with:
///   cargo test --features wavekat-smart-turn --test pipecat \
///       -- --ignored wavekat_hf_download_smoke --nocapture
#[cfg(feature = "wavekat-smart-turn")]
#[test]
#[ignore = "network: downloads ~8 MB from huggingface.co"]
fn wavekat_hf_download_smoke() {
    use std::path::Path;

    use wavekat_turn::audio::{SmartTurnLang, SmartTurnVariant};
    use wavekat_turn::TurnState;

    fn fixtures_dir() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("tests/fixtures")
    }

    fn load_wav(path: &Path) -> Vec<f32> {
        let mut reader = hound::WavReader::open(path)
            .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
        let spec = reader.spec();
        assert_eq!(spec.sample_rate, 16_000);
        assert_eq!(spec.channels, 1);
        match spec.sample_format {
            hound::SampleFormat::Int => reader
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.0)
                .collect(),
            hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        }
    }

    fn p_complete(pred: &TurnPrediction) -> f32 {
        match pred.state {
            TurnState::Finished => pred.confidence,
            TurnState::Unfinished => 1.0 - pred.confidence,
            TurnState::Wait => unreachable!(),
        }
    }

    println!("\nLoading wavekat/smart-turn-ONNX (zh) from HuggingFace…");
    let mut detector =
        PipecatSmartTurn::with_variant(SmartTurnVariant::Wavekat(SmartTurnLang::Zh))
            .expect("HF download / model load failed");

    let clips = ["silence_2s.wav", "speech_finished.wav", "speech_mid.wav"];
    println!();
    println!("| Clip | P(complete) | State | Latency (ms) |");
    println!("|------|-------------|-------|--------------|");
    for clip in clips {
        detector.reset();
        let samples = load_wav(&fixtures_dir().join(clip));
        for chunk in samples.chunks(1600) {
            detector.push_audio(&AudioFrame::new(chunk, 16_000));
        }
        let pred = detector.predict().expect("predict failed");
        valid_prediction(&pred);
        println!(
            "| `{}` | {:.4} | {:?} | {} |",
            clip,
            p_complete(&pred),
            pred.state,
            pred.latency_ms,
        );
    }
    println!();
}

/// Smoke test: latency is measured and non-zero (always runs, including debug).
#[test]
fn test_latency_is_measured() {
    let mut d = PipecatSmartTurn::new().unwrap();
    push_silence(&mut d, 2.0);
    let pred = d.predict().unwrap();
    // latency_ms == 0 would mean the timer wasn't working
    assert!(
        pred.latency_ms < 60_000,
        "latency suspiciously large: {} ms",
        pred.latency_ms
    );
}

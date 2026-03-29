//! Cross-validation accuracy test: Rust pipeline vs. Python reference.
//!
//! Verifies that our mel preprocessing and ONNX inference produce probabilities
//! within ±0.02 of the Python (Pipecat) reference for each fixture audio clip.
//!
//! Prerequisites:
//!   1. Run `python scripts/gen_reference.py` once to produce
//!      `tests/fixtures/reference.json` and `tests/fixtures/silence_2s.wav`.
//!   2. Commit those files alongside the WAV clips.
//!
//! Run individual regression tests: `cargo test --features pipecat --test accuracy`
//! Run the full report table:        `make accuracy`

use std::path::{Path, PathBuf};

const TOLERANCE: f32 = 0.02;

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap() // crates/
        .parent()
        .unwrap() // repo root
        .join("tests/fixtures")
}

#[derive(serde::Deserialize)]
struct RefEntry {
    file: String,
    probability: f32,
}

fn load_reference() -> Vec<RefEntry> {
    let path = fixtures_dir().join("reference.json");
    let json = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!(
            "missing {}: run `python scripts/gen_reference.py` first",
            path.display()
        )
    });
    serde_json::from_str(&json).expect("invalid reference.json")
}

fn reference_prob(entries: &[RefEntry], name: &str) -> f32 {
    entries
        .iter()
        .find(|e| e.file == name)
        .unwrap_or_else(|| panic!("no entry for '{}' in reference.json", name))
        .probability
}

/// Load a 16 kHz mono WAV and return samples as f32 in [-1, 1].
fn load_wav_f32(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("failed to open {}: {}", path.display(), e));
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "expected 16 kHz");
    assert_eq!(spec.channels, 1, "expected mono");
    match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    }
}

// ---------------------------------------------------------------------------
// Report row — one entry per (backend, clip)
// ---------------------------------------------------------------------------

struct Row {
    backend: &'static str,
    clip: String,
    python_prob: f32,
    rust_prob: f32,
}

impl Row {
    fn diff(&self) -> f32 {
        (self.rust_prob - self.python_prob).abs()
    }

    fn status(&self) -> &'static str {
        if self.diff() <= TOLERANCE { "PASS" } else { "FAIL" }
    }
}

// ---------------------------------------------------------------------------
// Pipecat backend
// ---------------------------------------------------------------------------

#[cfg(feature = "pipecat")]
mod pipecat {
    use wavekat_turn::audio::PipecatSmartTurn;
    use wavekat_turn::{AudioFrame, AudioTurnDetector, TurnPrediction, TurnState};

    use super::{fixtures_dir, load_wav_f32, reference_prob, RefEntry, Row, TOLERANCE};

    pub(super) fn rows(entries: &[RefEntry]) -> Vec<Row> {
        entries
            .iter()
            .map(|entry| {
                let samples = load_wav_f32(&fixtures_dir().join(&entry.file));
                let mut detector = PipecatSmartTurn::new().expect("failed to load model");
                for chunk in samples.chunks(1600) {
                    detector.push_audio(&AudioFrame::new(chunk, 16_000));
                }
                let pred = detector.predict().expect("predict failed");
                let rust_prob = raw_prob(&pred);
                Row {
                    backend: "pipecat",
                    clip: entry.file.clone(),
                    python_prob: entry.probability,
                    rust_prob,
                }
            })
            .collect()
    }

    fn raw_prob(pred: &TurnPrediction) -> f32 {
        match pred.state {
            TurnState::Finished => pred.confidence,
            TurnState::Unfinished => 1.0 - pred.confidence,
            TurnState::Wait => unreachable!(),
        }
    }

    pub(super) fn run_regression(clip: &str) {
        let entries = super::load_reference();
        let python_prob = reference_prob(&entries, clip);
        let row = rows(&[RefEntry {
            file: clip.to_string(),
            probability: python_prob,
        }])
        .remove(0);
        let diff = row.diff();
        assert!(
            diff <= TOLERANCE,
            "{clip}: rust={:.4} python={:.4} diff={diff:.4} (limit {TOLERANCE})",
            row.rust_prob,
            row.python_prob,
        );
    }

    #[test]
    fn test_accuracy_silence() {
        run_regression("silence_2s.wav");
    }

    #[test]
    fn test_accuracy_speech_finished() {
        run_regression("speech_finished.wav");
    }

    #[test]
    fn test_accuracy_speech_mid() {
        run_regression("speech_mid.wav");
    }
}

// Add future audio backends here:
//
// #[cfg(feature = "livekit-audio")]
// mod livekit_audio {
//     pub(super) fn rows(entries: &[super::RefEntry]) -> Vec<super::Row> { ... }
// }

// ---------------------------------------------------------------------------
// Accuracy report — prints a markdown table covering all enabled backends
// ---------------------------------------------------------------------------

/// Print a markdown table comparing Rust vs Python probabilities for all clips
/// across all enabled backends.
/// Run with: `make accuracy`
#[test]
#[ignore]
fn accuracy_report() {
    let entries = load_reference();
    let mut rows: Vec<Row> = Vec::new();

    #[cfg(feature = "pipecat")]
    rows.extend(pipecat::rows(&entries));

    let version = env!("CARGO_PKG_VERSION");
    println!();
    println!("BENCHMARK_VERSION={version}");
    println!();
    println!("| Backend | Clip | Python P(complete) | Rust P(complete) | Diff | Status |");
    println!("|---------|------|--------------------|------------------|------|--------|");
    for r in &rows {
        println!(
            "| {} | `{}` | {:.4} | {:.4} | {:.4} | {} |",
            r.backend,
            r.clip,
            r.python_prob,
            r.rust_prob,
            r.diff(),
            r.status(),
        );
    }
    println!();
}

//! Example: using TurnController for VAD-driven turn detection.
//!
//! Run with: `cargo run --features pipecat --example controller`
//!
//! Demonstrates the soft-reset flow using real WAV fixtures:
//!
//! 1. User speaks mid-sentence (speech_mid.wav) → Unfinished
//! 2. User continues speaking — soft reset keeps the buffer intact
//! 3. User finishes the sentence (speech_finished.wav) → Finished
//! 4. After assistant responds, hard reset starts a fresh turn

use std::path::Path;

use wavekat_turn::audio::PipecatSmartTurn;
use wavekat_turn::{AudioFrame, TurnController};

fn load_wav(path: &Path) -> Vec<f32> {
    let mut reader = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("failed to open {}: {}", path.display(), e));
    let spec = reader.spec();
    match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|s| s.unwrap() as f32 / 32768.0)
            .collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixtures = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests/fixtures");

    let speech_mid = load_wav(&fixtures.join("speech_mid.wav"));
    let speech_finished = load_wav(&fixtures.join("speech_finished.wav"));

    let detector = PipecatSmartTurn::new()?;
    let mut ctrl = TurnController::new(detector);

    // --- Speech A: user says something mid-sentence ---
    println!(">> VAD: speech started");
    ctrl.reset_if_finished(); // first speech → resets

    println!(">> Pushing speech_mid.wav (cut mid-sentence)");
    ctrl.push_audio(&AudioFrame::new(&speech_mid[..], 16_000));

    println!(">> VAD: speech ended");
    let result_a = ctrl.predict()?;
    println!(
        "   predict → {:?} (confidence: {:.3})",
        result_a.state, result_a.confidence
    );

    // --- Speech B: user continues speaking ---
    println!("\n>> VAD: speech started again");
    let did_reset = ctrl.reset_if_finished();
    println!(
        "   reset_if_finished → {}",
        if did_reset {
            "reset (turn was finished)"
        } else {
            "skipped (turn unfinished, keeping buffer)"
        }
    );

    println!(">> Pushing speech_finished.wav (complete sentence)");
    ctrl.push_audio(&AudioFrame::new(&speech_finished[..], 16_000));

    println!(">> VAD: speech ended");
    let result_b = ctrl.predict()?;
    println!(
        "   predict → {:?} (confidence: {:.3}, ran on A+B combined)",
        result_b.state, result_b.confidence
    );

    // --- New turn: after assistant responds ---
    println!("\n>> Assistant finished responding");
    ctrl.reset(); // hard reset for next turn
    println!("   hard reset, last_state: {:?}", ctrl.last_state());

    // --- Speech C: fresh turn ---
    println!("\n>> VAD: speech started (new turn)");
    ctrl.reset_if_finished(); // last_state is None → resets

    println!(">> Pushing speech_finished.wav");
    ctrl.push_audio(&AudioFrame::new(&speech_finished[..], 16_000));

    println!(">> VAD: speech ended");
    let result_c = ctrl.predict()?;
    println!(
        "   predict → {:?} (confidence: {:.3})",
        result_c.state, result_c.confidence
    );

    Ok(())
}

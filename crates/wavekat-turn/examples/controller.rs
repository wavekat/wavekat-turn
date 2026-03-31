//! Example: using TurnController for VAD-driven turn detection.
//!
//! Run with: `cargo run --features pipecat --example controller`
//!
//! This example simulates a VAD + turn detection flow where the user
//! speaks in two bursts ("I want to order..." then "...a pizza").
//! The controller's soft reset keeps the audio buffer intact between
//! the two bursts, so the second prediction sees the full context.

use wavekat_turn::audio::PipecatSmartTurn;
use wavekat_turn::{AudioFrame, TurnController};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let detector = PipecatSmartTurn::new()?;
    let mut ctrl = TurnController::new(detector);

    // Simulate 2 seconds of audio (silence for demo purposes).
    let audio_a = vec![0.0f32; 32_000];
    let audio_b = vec![0.0f32; 16_000];

    // --- Speech A: user starts talking ---
    println!(">> VAD: speech started");
    ctrl.reset_if_finished(); // first speech → resets

    println!(">> Pushing 2s of audio (speech A)");
    ctrl.push_audio(&AudioFrame::new(&audio_a[..], 16_000));

    println!(">> VAD: speech ended");
    let result_a = ctrl.predict()?;
    println!(
        "   predict → {:?} (confidence: {:.3})",
        result_a.state, result_a.confidence
    );

    // --- Speech B: user continues ---
    println!("\n>> VAD: speech started again");
    let did_reset = ctrl.reset_if_finished();
    println!(
        "   reset_if_finished → {}",
        if did_reset { "reset" } else { "skipped (turn unfinished)" }
    );

    println!(">> Pushing 1s of audio (speech B)");
    ctrl.push_audio(&AudioFrame::new(&audio_b[..], 16_000));

    println!(">> VAD: speech ended");
    let result_b = ctrl.predict()?;
    println!(
        "   predict → {:?} (confidence: {:.3}, ran on A+B combined)",
        result_b.state, result_b.confidence
    );

    // --- New turn: after assistant responds ---
    println!("\n>> Assistant finished responding, hard reset");
    ctrl.reset();
    println!("   last_state: {:?}", ctrl.last_state());

    // --- Speech C: new turn ---
    println!("\n>> VAD: speech started (new turn)");
    ctrl.reset_if_finished(); // last_state is None → resets

    println!(">> Pushing 1s of audio (speech C)");
    ctrl.push_audio(&AudioFrame::new(&audio_b[..], 16_000));

    println!(">> VAD: speech ended");
    let result_c = ctrl.predict()?;
    println!(
        "   predict → {:?} (confidence: {:.3})",
        result_c.state, result_c.confidence
    );

    Ok(())
}

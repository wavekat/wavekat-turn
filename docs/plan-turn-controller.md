# Plan: TurnController Wrapper

**Status:** Not started
**Date:** 2026-03-31

---

## Problem

The `AudioTurnDetector` trait documents a simple flow:

1. Every audio chunk → `push_audio`
2. VAD fires "speech started" → `reset`
3. VAD fires "speech stopped" → `predict`

This works for the basic case, but breaks when the user continues speaking after a
brief pause — a common pattern in natural conversation (e.g. "I want to order... um...
a pizza").

### What goes wrong

Consider this sequence:

```
VAD speech start  → reset()           buffer cleared
  (push_audio)                        buffer has speech A
VAD speech end    → predict()         → Unfinished
VAD speech start  → reset()           ← WRONG: clears speech A
  (push_audio)                        buffer has speech B only
VAD speech end    → predict()         runs on B alone, missing context
```

The Pipecat Smart Turn documentation explicitly says:

> If additional speech is detected from the user before Smart Turn has finished
> executing, re-run Smart Turn on the entire turn recording, including the new audio,
> rather than just the new segment. Smart Turn works best when given sufficient context,
> and is not designed to run on very short audio segments.

The correct behavior is to **skip the reset** when the previous prediction was
`Unfinished`, so the buffer accumulates across the full turn:

```
VAD speech start  → reset()           buffer cleared (first speech)
  (push_audio)                        buffer has speech A
VAD speech end    → predict()         → Unfinished
VAD speech start  → DON'T reset       buffer keeps speech A
  (push_audio)                        buffer has speech A + B
VAD speech end    → predict()         runs on A+B combined ✓
```

### Why this doesn't belong in the trait

The `AudioTurnDetector` trait is the right abstraction for backend authors — it's
minimal, and `reset()` is a clean primitive ("clear everything"). The soft-reset
decision depends on tracking the last prediction state, which is orchestration logic.

Every orchestrator would have to re-implement this same logic. As a library, we should
provide a helper that does it correctly out of the box.

---

## Solution: `TurnController<T>`

A generic wrapper around any `AudioTurnDetector` that tracks prediction state and
provides convenience methods.

```rust
pub struct TurnController<T: AudioTurnDetector> {
    inner: T,
    last_state: Option<TurnState>,
}
```

### API

```rust
impl<T: AudioTurnDetector> TurnController<T> {
    /// Create a new controller wrapping the given detector.
    pub fn new(inner: T) -> Self;

    /// Feed audio into the detector.
    pub fn push_audio(&mut self, frame: &AudioFrame);

    /// Run prediction on buffered audio.
    /// Tracks the result state internally for `reset_if_finished`.
    pub fn predict(&mut self) -> Result<TurnPrediction, TurnError>;

    /// Hard reset — always clears the buffer. Use when you know a new turn
    /// is starting (e.g. after the assistant finishes responding).
    pub fn reset(&mut self);

    /// Soft reset — clears the buffer only if the last prediction was
    /// `Finished` (or no prediction has been made yet). Returns whether
    /// a reset actually occurred.
    ///
    /// Call this on VAD speech-start when you don't know whether the user
    /// is continuing the same turn or starting a new one.
    pub fn reset_if_finished(&mut self) -> bool;

    /// Returns the state from the last `predict()` call, or `None` if
    /// no prediction has been made since the last reset.
    pub fn last_state(&self) -> Option<TurnState>;

    /// Unwrap the controller, returning the inner detector.
    pub fn into_inner(self) -> T;
}
```

### Usage

```rust
let detector = PipecatSmartTurn::new()?;
let mut ctrl = TurnController::new(detector);

// Audio arrives continuously
ctrl.push_audio(&frame);

// VAD speech start — soft reset (keeps buffer if turn was unfinished)
ctrl.reset_if_finished();

// VAD speech end — predict
let result = ctrl.predict()?;
match result.state {
    TurnState::Finished   => { /* hand off to LLM */ }
    TurnState::Unfinished => { /* wait for more speech */ }
}

// After assistant finishes responding — hard reset for next turn
ctrl.reset();
```

### Scenario walkthrough

```rust
// Speech A — user says "I want to order..."
ctrl.reset_if_finished();          // no prior prediction → resets ✓
ctrl.push_audio(&speech_a);
let a = ctrl.predict()?;           // → Unfinished

// Speech B — user continues "...a pizza"
ctrl.reset_if_finished();          // last was Unfinished → NO reset ✓
ctrl.push_audio(&speech_b);
let b = ctrl.predict()?;           // runs on A+B combined → Finished ✓

// Speech C — new conversation turn
ctrl.reset();                      // hard reset after assistant responded
ctrl.push_audio(&speech_c);
let c = ctrl.predict()?;           // runs on C only ✓
```

---

## Design decisions

### Why `TurnController` and not a trait method

- Rust traits can't have fields, so every implementor would duplicate the
  `last_state` tracking boilerplate.
- The soft-reset logic is identical across all backends — it only depends on
  `TurnState`, not on backend internals.
- A wrapper keeps the trait minimal for backend authors while giving orchestrators
  a batteries-included API.

### Why `reset_if_finished` returns `bool`

The orchestrator may want to know whether a reset occurred — e.g. for logging,
or to adjust behavior (start a new transcript vs. append to existing).

### Why keep `reset()` on the controller

Hard reset is still needed for cases the controller can't infer:
- After the assistant finishes responding (new conversation turn).
- Manual override / error recovery.
- First initialization.

`reset_if_finished()` is the default for VAD speech-start events.
`reset()` is for explicit turn boundaries the orchestrator controls.

---

## Future possibilities

These are not part of the initial implementation but the `TurnController` is a
natural place to add them later:

- **Min audio guard** — `predict()` returns early if the buffer is too short to
  produce a meaningful prediction, avoiding wasted inference on tiny audio segments.
- **Configurable threshold** — override the default 0.5 probability threshold
  without modifying the detector.
- **Prediction history** — track recent predictions for debugging and logging.

---

## File placement

```
src/
├── lib.rs                 — existing traits (unchanged)
├── controller.rs          — TurnController<T>     ← NEW
├── audio/
│   └── pipecat.rs         — PipecatSmartTurn (unchanged)
└── ...
```

Re-export from `lib.rs`:

```rust
mod controller;
pub use controller::TurnController;
```

---

## Tests

| Test | What it checks |
|------|---------------|
| `reset_if_finished_resets_on_first_call` | No prior prediction → resets |
| `reset_if_finished_skips_after_unfinished` | Last predict was Unfinished → no reset |
| `reset_if_finished_resets_after_finished` | Last predict was Finished → resets |
| `hard_reset_always_clears` | `reset()` clears regardless of last state |
| `last_state_tracks_predictions` | `last_state()` returns correct value after predict/reset |
| `predict_accumulates_across_soft_reset` | Buffer preserved when soft reset skips → predict uses full audio |

<p align="center">
  <a href="https://github.com/wavekat/wavekat-turn">
    <img src="https://github.com/wavekat/wavekat-brand/raw/main/assets/banners/wavekat-turn-narrow.svg" alt="WaveKat Turn">
  </a>
</p>

[![Crates.io](https://img.shields.io/crates/v/wavekat-turn.svg)](https://crates.io/crates/wavekat-turn)
[![docs.rs](https://docs.rs/wavekat-turn/badge.svg)](https://docs.rs/wavekat-turn)

Unified turn detection for voice pipelines, wrapping multiple open-source
models behind common Rust traits. Same pattern as
[wavekat-vad](https://github.com/wavekat/wavekat-vad).

> [!WARNING]
> Early development. API may change between minor versions.

## Backends

| Backend | Feature flag | Input | Model size | Inference | License |
|---------|-------------|-------|------------|-----------|---------|
| [Pipecat Smart Turn v3](https://github.com/pipecat-ai/smart-turn) | `pipecat` | Audio (16 kHz PCM) | ~8 MB (int8 ONNX) | ~12 ms CPU | BSD 2-Clause |
| [LiveKit Turn Detector](https://github.com/livekit/turn-detector) | `livekit` | Text (ASR transcript) | ~400 MB (ONNX) | ~25 ms CPU | LiveKit Model License |

## Quick Start

```sh
cargo add wavekat-turn --features pipecat
```

Use `TurnController` to wrap any detector with automatic state tracking:

```rust
use wavekat_turn::{TurnController, TurnState};
use wavekat_turn::audio::PipecatSmartTurn;

let detector = PipecatSmartTurn::new()?;
let mut ctrl = TurnController::new(detector);

// Feed audio continuously
ctrl.push_audio(&audio_frame);

// VAD speech start — soft reset (keeps buffer if turn was unfinished)
ctrl.reset_if_finished();

// VAD speech end — predict
let prediction = ctrl.predict()?;
match prediction.state {
    TurnState::Finished   => { /* user is done, send to LLM */ }
    TurnState::Unfinished => { /* keep listening */ }
    TurnState::Wait       => { /* user asked AI to hold */ }
}

// After assistant finishes responding — hard reset
ctrl.reset();
```

Or the text-based detector directly:

```rust
use wavekat_turn::{TextTurnDetector, TurnState};
use wavekat_turn::text::LiveKitEou;

let mut detector = LiveKitEou::new()?;

let prediction = detector.predict_text("I was wondering if", &context)?;
assert_eq!(prediction.state, TurnState::Unfinished);
```

See [`examples/controller.rs`](crates/wavekat-turn/examples/controller.rs) for a
full walkthrough with real audio.

## Architecture

Two trait families cover the two input modalities:

- **`AudioTurnDetector`** -- operates on raw audio frames (no ASR needed)
- **`TextTurnDetector`** -- operates on ASR transcript text with optional conversation context

`TurnController` wraps any `AudioTurnDetector` and adds orchestration helpers
like soft-reset (preserves buffer when the user pauses mid-sentence).

```
wavekat-vad   -->  "is someone speaking?"
wavekat-turn  -->  "are they done speaking?"
     |                   |
     v                   v
wavekat-voice -->  orchestrates VAD + turn + ASR + LLM + TTS
```

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `pipecat` | off | Pipecat Smart Turn v3 audio backend (requires `ort`, `ndarray`) |
| `livekit` | off | LiveKit text-based backend (requires `ort`, `ndarray`) |

## Important Notes

- **8 kHz telephony audio must be upsampled to 16 kHz** before passing to
  audio-based detectors. Smart Turn v3 silently produces incorrect results
  at 8 kHz.
- Text-based detectors depend on ASR transcript quality. Pair with a
  streaming ASR provider for best results.

## Accuracy

Cross-validated against the original Python (Pipecat) pipeline on three fixture clips.
Tolerance: ±0.02 probability.

<!-- benchmark-table-start -->
<!-- benchmark-table-end -->

Run locally with `make accuracy`. See [`scripts/README.md`](scripts/README.md) for how to regenerate the Python reference.

## License

Licensed under [Apache 2.0](LICENSE).

Copyright 2026 WaveKat.

### Acknowledgements

- [Pipecat Smart Turn](https://github.com/pipecat-ai/smart-turn) by Daily (BSD 2-Clause)
- [LiveKit Turn Detector](https://github.com/livekit/turn-detector) by LiveKit (LiveKit Model License)

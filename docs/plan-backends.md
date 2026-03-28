# Plan: Implement Turn Detection Backends

**Status:** In progress
**Date:** 2026-03-28

---

## Decisions made

- **Pipecat first.** Implement `PipecatSmartTurn` (audio, ~8 MB) before `LiveKitEou` (text,
  ~400 MB). Smaller model, faster to iterate.
- **Follow wavekat-vad pattern.** Build-time model download via `build.rs` + `include_bytes!()`,
  same env-var overrides (`*_MODEL_PATH`, `*_MODEL_URL`).
- **Turn logic stays here.** The lab (`wavekat-lab`) calls these backends as a library consumer.
  No turn logic lives in the lab.

---

## Current state

Both backends are stubs with `todo!()` — the crate compiles but cannot run inference.

```
src/
├── lib.rs              — traits: AudioTurnDetector, TextTurnDetector, TurnPrediction, TurnState
├── error.rs            — TurnError: BackendError, InvalidInput, ModelNotLoaded
├── audio/
│   ├── mod.rs
│   └── pipecat.rs      — PipecatSmartTurn (stub)
└── text/
    ├── mod.rs
    └── livekit.rs      — LiveKitEou (stub)
```

No `build.rs` yet. No tests.

---

## Trait API (stable, do not change)

```rust
pub trait AudioTurnDetector: Send + Sync {
    fn push_audio(&mut self, frame: &AudioFrame);   // 16 kHz mono f32
    fn predict(&mut self) -> Result<TurnPrediction, TurnError>;
    fn reset(&mut self);
}

pub trait TextTurnDetector: Send + Sync {
    fn predict_text(&mut self, transcript: &str, context: &[ConversationTurn])
        -> Result<TurnPrediction, TurnError>;
    fn reset(&mut self);
}
```

`TurnPrediction` — `{ state: TurnState, confidence: f32, latency_ms: u64 }`
`TurnState` — `Finished | Unfinished | Wait`

---

## Phase 1 — Research (prerequisite)

Before writing any code, pin down the model specifics:

1. **Model source.** Find the official Pipecat Smart Turn v3 ONNX download URL (Pipecat GitHub
   releases or Hugging Face). Confirm license (BSD 2-Clause noted in stub comments).

2. **Input/output tensor shapes.** Load the model in a scratch script or `netron` and record:
   - Input tensor: name, shape, dtype
   - Output tensor: name(s), shape, dtype
   - Whether output is a single confidence float or logits for [Finished, Unfinished, Wait]

3. **Mel-feature spec.** Confirm what preprocessing the model expects:
   - Frame size + hop length
   - Number of mel bins (Whisper uses 80)
   - Frequency range
   - Mel scale formula (HTK vs Kaldi)
   - Whether pre-emphasis is applied

4. **Audio buffer length.** Stub says "up to 8 seconds" — confirm from model input shape.

Document findings as comments in `pipecat.rs` before implementation.

---

## Phase 2 — Build system

Create `crates/wavekat-turn/build.rs` following the wavekat-vad pattern:

- Download Smart Turn v3 ONNX to `OUT_DIR` at build time
- SHA-256 verification
- Env-var overrides:
  - `PIPECAT_SMARTTURN_MODEL_PATH` — use a local file instead of downloading
  - `PIPECAT_SMARTTURN_MODEL_URL` — override download URL
- Docs.rs guard: write a zero-byte placeholder when `DOCS_RS=1`

Add to `Cargo.toml`:
```toml
[package]
build = "build.rs"

[build-dependencies]
ureq = { version = "3", features = ["tls"] }
```

---

## Phase 3 — PipecatSmartTurn implementation

Fill in `src/audio/pipecat.rs`:

**Struct:**
```rust
pub struct PipecatSmartTurn {
    session: Session,
    ring_buffer: VecDeque<f32>,  // 8s × 16kHz = 128k samples
    // mel extractor fields TBD from Phase 1 research
}
```

**`new()`** — load model via `include_bytes!(concat!(env!("OUT_DIR"), "/..."))`,
create `ort::Session`, initialize ring buffer.

**`push_audio()`** — validate sample rate (16 kHz), convert i16→f32 if needed,
append to ring buffer (evict oldest when over capacity).

**`predict()`** — snapshot ring buffer, pad/truncate to model's expected length,
extract mel features, build ndarray input tensor, `session.run(...)`, parse output,
record `Instant` before/after for `latency_ms`.

**`reset()`** — `ring_buffer.clear()`.

Reference implementations:
- `wavekat-vad/src/backends/silero.rs` — ONNX session + state management
- `wavekat-vad/src/backends/onnx.rs` — session builder helper
- `wavekat-vad/src/backends/firered/fbank.rs` — mel filterbank (adapt if spec matches)

---

## Phase 4 — Tests

Add `tests/pipecat.rs` (integration tests under `#[cfg(feature = "pipecat")]`):

- `test_new_loads_model` — `PipecatSmartTurn::new()` succeeds
- `test_predict_silence` — feed 2s of zeros, expect low confidence
- `test_predict_finished` — feed known-good finished-turn audio (WAV fixture), expect
  `TurnState::Finished` with confidence > 0.7
- `test_reset_clears_buffer` — push audio, reset, predict on empty buffer returns low confidence
- `test_rtf` — assert `latency_ms` < 50 ms (well under the ~12 ms target with headroom for CI)

Add a small WAV fixture (`tests/fixtures/finished_turn.wav`) for the audio test cases.

---

## Phase 5 — LiveKitEou (deferred)

Implement `src/text/livekit.rs` after Pipecat is proven end-to-end.

Different approach — text model using a tokenizer:
- Model: distilled Qwen2.5-0.5B ONNX (~400 MB), LiveKit Model License
- Input: tokenized transcript + conversation context
- Likely needs a tokenizer crate (e.g. `tokenizers` from HuggingFace)

Open questions before starting:
1. Confirm model URL and whether `tokenizers` crate is acceptable (it's a large dep)
2. Confirm exact input format the model expects for transcript + context

---

## Open questions

1. **Smart Turn v3 model URL** — not yet confirmed (needed for Phase 2)
2. **Exact input tensor shape** — need to inspect the model (needed for Phase 3)
3. **Mel-feature spec** — need to confirm to avoid silent preprocessing mismatch
4. **LiveKit tokenizer strategy** — `tokenizers` crate vs. manual BPE (needed before Phase 5)

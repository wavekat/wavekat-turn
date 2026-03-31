# Plan: Implement Turn Detection Backends

**Status:** Phase 1–4 complete
**Date:** 2026-03-28

---

## Decisions made

- **Pipecat first.** Implement `PipecatSmartTurn` (audio, ~8 MB) before `LiveKitEou` (text,
  ~400 MB). Smaller model, faster to iterate.
- **Follow wavekat-vad pattern.** Build-time model download via `build.rs` + `include_bytes!()`,
  same env-var overrides (`*_MODEL_PATH`, `*_MODEL_URL`).
- **Turn logic stays here.** The lab (`wavekat-lab`) calls these backends as a library consumer.
  No turn logic lives in the lab.
- **Model loading strategy by size.**
  - **< ~30 MB → embed** with `include_bytes!()`. Binary size is acceptable; zero runtime setup.
    Pipecat (8 MB) uses this path.
  - **≥ ~30 MB → runtime load** from disk. Embedding would bloat the binary unacceptably.
    Future large-model backends must use this path (see "Out of scope" section).
- **`from_file()` constructor on all backends.** Even embedded-model backends expose a
  `from_file(path)` constructor so users can substitute custom or fine-tuned weights, and to
  establish the pattern that future large-model backends will use as their primary constructor.

---

## Current state

`PipecatSmartTurn` is fully implemented and all integration tests pass.
`TurnController` wraps any `AudioTurnDetector` with state tracking and soft-reset.
`LiveKitEou` remains a stub (out of scope for this branch).

```
src/
├── lib.rs              — traits: AudioTurnDetector, TextTurnDetector, TurnPrediction, TurnState
├── controller.rs       — TurnController<T> orchestration wrapper
├── error.rs            — TurnError: BackendError, InvalidInput, ModelNotLoaded
├── onnx.rs             — shared session_from_file / session_from_memory helpers
├── audio/
│   ├── mod.rs
│   └── pipecat.rs      — PipecatSmartTurn (complete)
└── text/
    ├── mod.rs
    └── livekit.rs      — LiveKitEou (stub, out of scope)
build.rs                — downloads smart-turn-v3.2-cpu.onnx at build time
examples/
└── controller.rs       — TurnController usage with real WAV fixtures
tests/
├── controller.rs       — 7 TurnController tests (mock detector)
└── pipecat.rs          — 9 integration tests (all pass)
```

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

## Phase 1 — Research ✅

**Done.** Findings pinned in `src/audio/pipecat.rs` module-level comments.

| Item | Finding |
|------|---------|
| Model URL | `https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx` |
| Input tensor | `input_features`, shape `[B, 80, 800]`, float32 |
| Output tensor | `logits`, shape `[B, 1]`, float32 — sigmoid P(turn complete), NOT raw logits |
| Mel scale | **Slaney** (NOT HTK); `norm="slaney"` |
| n_fft / hop | 400 / 160 samples (25 ms / 10 ms at 16 kHz) |
| Mel bins | 80; frequency range 0–8 000 Hz |
| Window | Periodic Hann (`torch.hann_window(400, periodic=True)`) |
| Pre-emphasis | None |
| Log norm | `log10`, clamp `[max−8, ∞]`, then `(x + 4) / 4` |
| Audio buffer | 8 s = 128 000 samples; front-pad shorter, keep last 8 s for longer |
| License | BSD 2-Clause |

---

## Phase 2 — Build system ✅

**Done.**

- `build.rs` downloads `smart-turn-v3.2-cpu.onnx` to `OUT_DIR` with version-based caching
- Env-var overrides: `PIPECAT_SMARTTURN_MODEL_PATH`, `PIPECAT_SMARTTURN_MODEL_URL`
- Docs.rs guard writes a zero-byte placeholder when `DOCS_RS=1`
- `Cargo.toml`: `build = "build.rs"`, `ureq` as optional build-dep activated by `pipecat` feature

Note: SHA-256 verification was omitted in favour of version-based caching (same as wavekat-vad).

---

## Phase 3 — PipecatSmartTurn implementation ✅

**Done.** `src/audio/pipecat.rs` and `src/onnx.rs` written and compiling.

Key implementation decisions:
- `MelExtractor` precomputes the Slaney filterbank matrix and Hann window once at construction;
  reuses FFT plan and scratch buffers across calls
- Center-pad (`N_FFT/2` zeros each side) replicates librosa `center=True` STFT, producing
  exactly 800 frames from 128 000 samples
- `push_audio` silently drops frames with wrong sample rate (no return value in trait)
- `ndarray = "0.17"` required to match `ort`'s ndarray feature version

---

## Phase 4 — Tests ✅

**Done.** `tests/pipecat.rs` with 9 integration tests, all passing:

| Test | What it checks |
|------|---------------|
| `test_new_loads_model` | `new()` succeeds |
| `test_from_file_loads_model` | `from_file()` succeeds with a valid path |
| `test_predict_returns_valid_output` | confidence ∈ [0, 1] |
| `test_predict_with_empty_buffer` | empty buffer inference succeeds |
| `test_push_audio_wrong_sample_rate_is_ignored` | 8 kHz frame is dropped |
| `test_reset_clears_buffer` | state after reset matches fresh instance |
| `test_ring_buffer_caps_at_8_seconds` | 10 s of audio doesn't panic |
| `test_multiple_predicts_are_deterministic` | same buffer → same output |
| `test_latency_under_50ms` | RTF < 50 ms (release builds only) |

Note: the current tests do not cross-validate against the Python reference implementation.
That is tracked in **[`plan-accuracy.md`](plan-accuracy.md)**.

---

## Open questions

All research questions from Phases 1–3 are resolved. No blocking open questions remain
for this branch.

---

## Out of scope — LiveKitEou

> **Not part of this branch.** Will be implemented in a dedicated feature branch.

Key notes to carry forward:

- Model: distilled Qwen2.5-0.5B ONNX (~400 MB), LiveKit Model License
- Input: tokenized transcript + conversation context
- At 400 MB, `include_bytes!()` is not viable — needs a runtime-load strategy (build.rs
  downloads to a user cache dir, binary loads from disk via `from_file()`)
- The `from_file()` constructor established on `PipecatSmartTurn` in Phase 3 gives LiveKit
  the same public API shape to follow
- Open questions for that branch: model URL, `tokenizers` crate acceptability, exact input
  format, CI/CD cache-dir strategy

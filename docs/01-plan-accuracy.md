# Plan: Cross-Validate Rust Implementation Against Python Reference

**Status:** Not started
**Date:** 2026-03-28

---

## Goal

Verify that our Rust mel preprocessing and ONNX inference pipeline produces probabilities that
match the original Pipecat Python implementation within a tight tolerance. This catches any
silent preprocessing mismatch (wrong mel scale, wrong window, wrong padding, wrong normalization)
that unit tests cannot detect because they only check output shape and range.

---

## Why this matters

The Rust implementation re-implements `WhisperFeatureExtractor` from scratch. Any divergence
in the mel filterbank, Hann window, STFT center-padding, or log-normalization will silently
shift probabilities. A 5% probability shift near the 0.5 threshold flips a turn decision.
The only way to be confident the two pipelines agree is to feed them identical audio and
compare outputs numerically.

---

## Decisions made

- **Python script generates reference data once.** Output is committed as a JSON fixture
  (`tests/fixtures/reference.json`) so the Rust accuracy test has no Python runtime dependency.
- **Tolerance: ±0.02 probability.** The model uses float32 throughout; numerical differences
  from the Rust FFT vs NumPy FFT should be well under this. If a case fails, it signals a
  real preprocessing bug, not floating-point noise.
- **Three fixture audio clips.** Enough to cover the key behavioral regions without large
  binary assets. Total fixture size should stay under ~500 KB.

---

## Fixture audio clips

| File | Content | Expected region |
|------|---------|-----------------|
| `tests/fixtures/silence_2s.wav` | 2 s of zeros at 16 kHz | Low P(complete) — no speech |
| `tests/fixtures/speech_finished.wav` | Real or synthetic utterance that ends cleanly | High P(complete) |
| `tests/fixtures/speech_mid.wav` | Real or synthetic utterance cut mid-word | Low P(complete) |

WAV format: 16 kHz, mono, 16-bit PCM (hound-compatible).

For `speech_finished.wav` and `speech_mid.wav`, use short clips (1–3 s) from a freely
licensed speech corpus, or generate synthetic speech with a TTS tool. Commit the WAVs
directly — they are small enough.

---

## Phase 1 — Python reference script

Create `scripts/gen_reference.py`.

**Dependencies** (not added to the crate — Python only):
```
pip install pipecat-ai transformers onnxruntime numpy soundfile
```

**What it does:**
1. Downloads `smart-turn-v3.2-cpu.onnx` if not already present (same URL as build.rs)
2. For each WAV in `tests/fixtures/`:
   - Loads audio via `soundfile` as float32 at 16 kHz
   - Runs `WhisperFeatureExtractor(chunk_length=8)` to get `input_features`
   - Runs `ort.InferenceSession` on the ONNX model
   - Records `{ "file": "...", "probability": <float> }`
3. Writes `tests/fixtures/reference.json`

**Re-run when:**
- A fixture WAV changes
- The model version changes (bump `MODEL_VERSION` in `build.rs` at the same time)

---

## Phase 2 — WAV fixtures

Generate or source the three WAV clips and commit them to `tests/fixtures/`.

For `silence_2s.wav`:
```python
import numpy as np, soundfile as sf
sf.write("tests/fixtures/silence_2s.wav", np.zeros(32000, dtype=np.float32), 16000)
```

For speech clips, options in order of preference:
1. Record 2–3 s clips specifically for this test
2. Use a clip from [CMU Arctic](http://www.festvox.org/cmu_arctic/) or
   [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) (both public domain / CC0)
3. Generate with `piper` TTS (Apache 2.0)

---

## Phase 3 — Run the Python script and commit reference.json

```bash
python scripts/gen_reference.py
```

Inspect the output — confirm the probabilities make sense (silence ≈ low, finished ≈ high).
Commit `tests/fixtures/reference.json` alongside the WAV files.

`reference.json` format:
```json
[
  { "file": "silence_2s.wav",      "probability": 0.03 },
  { "file": "speech_finished.wav", "probability": 0.91 },
  { "file": "speech_mid.wav",      "probability": 0.08 }
]
```

---

## Phase 4 — Rust accuracy test

Add `tests/accuracy.rs` (under `#[cfg(feature = "pipecat")]`).

**What it does:**
1. Reads `tests/fixtures/reference.json` at test time
2. For each entry, loads the corresponding WAV with `hound`
3. Pushes all audio frames through `PipecatSmartTurn`
4. Calls `predict()` and reads the raw probability
5. Asserts `|rust_prob - python_prob| <= TOLERANCE` (0.02)

```rust
const TOLERANCE: f32 = 0.02;
```

**Getting the raw probability out of `TurnPrediction`:**
`TurnPrediction.confidence` is already the raw sigmoid value (we set it to `probability` for
`Finished` and `1.0 - probability` for `Unfinished`). To recover the original probability:

```rust
let raw_prob = match pred.state {
    TurnState::Finished   => pred.confidence,
    TurnState::Unfinished => 1.0 - pred.confidence,
    TurnState::Wait       => unreachable!(),
};
```

Alternatively, expose `raw_probability: f32` directly on `TurnPrediction` — see open questions.

**Test names:**
- `test_accuracy_silence`
- `test_accuracy_speech_finished`
- `test_accuracy_speech_mid`

---

## Open questions

1. **Expose raw probability on `TurnPrediction`?**
   Currently the struct only has `confidence` which loses the original P(complete) for
   `Unfinished` cases. Options:
   - Add `raw_probability: f32` field to `TurnPrediction` (cleaner, but changes the public API)
   - Reconstruct from `(state, confidence)` in the test (works, but fragile)
   Resolve before starting Phase 4.

2. **Speech fixture source.** Decide on LJ Speech clips or recorded clips before Phase 2.
   LJ Speech is easiest (download a sentence, trim to 2–3 s). Record the chosen file name
   and source URL in a comment in `scripts/gen_reference.py`.

3. **CI integration.** The accuracy test needs the WAV fixtures and `reference.json` committed
   to the repo. Confirm the total asset size is acceptable before merging.

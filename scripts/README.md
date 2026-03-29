# scripts/

## gen_reference.py

Generates `tests/fixtures/reference.json` — the Python-side reference probabilities
used by the Rust accuracy test (`make accuracy`).

### Setup

```sh
python3 -m venv scripts/.venv
scripts/.venv/bin/pip install transformers onnxruntime numpy soundfile
```

### Run

```sh
scripts/.venv/bin/python3 scripts/gen_reference.py
```

### Re-run when

- A fixture WAV changes (`tests/fixtures/*.wav`)
- The model version changes (bump `MODEL_VERSION` in `build.rs` at the same time)

### What it produces

| File | Description |
|------|-------------|
| `tests/fixtures/silence_2s.wav` | 2 s of zeros at 16 kHz (generated if missing) |
| `tests/fixtures/reference.json` | P(complete) for each fixture clip |

Commit both files after re-running.

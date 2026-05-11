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
| `tests/fixtures/reference.json` | P(complete) for each `(backend, clip)` pair |

Each entry in `reference.json` is keyed by `backend` so multiple Smart Turn
variants can coexist:

- `pipecat` — upstream Pipecat Smart Turn v3 on the English fixtures, plus the
  same model run on the Mandarin fixtures as a cross-lingual baseline.
- `wavekat-zh` — WaveKat zh fine-tune of Smart Turn, only run on the
  Mandarin fixtures (`zh_*.wav`).

The Rust accuracy test (`make accuracy`) filters rows by backend at compile
time — feature `wavekat-smart-turn` enables the second group.

Commit both files after re-running.

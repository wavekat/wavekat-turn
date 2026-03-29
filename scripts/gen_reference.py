#!/usr/bin/env python3
"""Generate reference probabilities from the Pipecat Python pipeline.

Outputs tests/fixtures/reference.json for use in the Rust accuracy test.

Usage:
    pip install transformers onnxruntime numpy soundfile
    python scripts/gen_reference.py

Re-run when:
  - A fixture WAV changes
  - The model version changes (bump MODEL_VERSION in build.rs at the same time)

Speech fixture source:
  speech_finished.wav and speech_mid.wav are original recordings of:
    "Wavekat knows when you've finished speaking."
  recorded at 16 kHz mono 16-bit PCM.
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from transformers import WhisperFeatureExtractor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
SCRIPTS = REPO_ROOT / "scripts"

MODEL_URL = "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
MODEL_VERSION = "v3.2-cpu"
MODEL_CACHE = SCRIPTS / f"smart-turn-{MODEL_VERSION}.onnx"

SAMPLE_RATE = 16_000
BUFFER_SAMPLES = 128_000  # 8 seconds at 16 kHz (matches Rust ring buffer)

CLIPS = ["silence_2s.wav", "speech_finished.wav", "speech_mid.wav"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_model() -> Path:
    if MODEL_CACHE.exists():
        return MODEL_CACHE
    print(f"Downloading model from {MODEL_URL} ...", flush=True)
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_CACHE)
    print(f"Saved to {MODEL_CACHE}", flush=True)
    return MODEL_CACHE


def ensure_silence() -> None:
    path = FIXTURES / "silence_2s.wav"
    if not path.exists():
        print("Generating silence_2s.wav ...", flush=True)
        FIXTURES.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), np.zeros(32_000, dtype=np.float32), SAMPLE_RATE, subtype="PCM_16")


def load_audio(path: Path) -> np.ndarray:
    """Load WAV as mono float32 at 16 kHz, front-padded to 8 s."""
    audio, sr = sf.read(str(path), dtype="float32")
    assert sr == SAMPLE_RATE, f"{path.name}: expected {SAMPLE_RATE} Hz, got {sr}"
    assert audio.ndim == 1, f"{path.name}: expected mono audio"
    # Front-pad with zeros to match Rust ring-buffer behaviour (shorter → zeros at front)
    if len(audio) < BUFFER_SAMPLES:
        audio = np.pad(audio, (BUFFER_SAMPLES - len(audio), 0))
    else:
        audio = audio[-BUFFER_SAMPLES:]
    return audio


def infer(audio: np.ndarray, session, extractor) -> float:
    """Run the Pipecat pipeline on audio, return P(complete)."""
    features = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
    input_features = features["input_features"].astype(np.float32)  # [1, 80, 800]
    outputs = session.run(None, {"input_features": input_features})
    return float(np.squeeze(outputs[0]))  # already a sigmoid probability in [0, 1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("ERROR: onnxruntime not installed.  Run: pip install onnxruntime", file=sys.stderr)
        sys.exit(1)

    ensure_silence()
    model_path = ensure_model()

    extractor = WhisperFeatureExtractor(chunk_length=8)
    session = ort.InferenceSession(str(model_path))

    results = []
    for name in CLIPS:
        path = FIXTURES / name
        if not path.exists():
            print(f"ERROR: missing fixture {path}", file=sys.stderr)
            sys.exit(1)
        audio = load_audio(path)
        prob = infer(audio, session, extractor)
        print(f"  {name}: probability = {prob:.4f}")
        results.append({"file": name, "probability": round(prob, 6)})

    out_path = FIXTURES / "reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

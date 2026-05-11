#!/usr/bin/env python3
"""Generate reference probabilities from the Python pipelines.

Outputs tests/fixtures/reference.json for use in the Rust accuracy test.
Each entry is keyed by ``(backend, file)``; the Rust test filters by enabled
backend at compile time.

Usage:
    pip install transformers onnxruntime numpy soundfile
    python scripts/gen_reference.py

Re-run when:
  - A fixture WAV changes
  - A model version changes (bump MODEL_VERSION constants below + build.rs)

Backends covered:
  - ``pipecat``     — upstream Pipecat Smart Turn v3.2-cpu, scored on
                       silence_2s / speech_finished / speech_mid (English).
  - ``wavekat-zh``  — WaveKat zh fine-tune of Smart Turn, scored on
                       zh_speech_finished / zh_speech_finished_short /
                       zh_speech_mid (Mandarin).

Speech fixture source (English):
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

PIPECAT_MODEL_URL = (
    "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx"
)
PIPECAT_MODEL_VERSION = "v3.2-cpu"
PIPECAT_MODEL_CACHE = SCRIPTS / f"smart-turn-{PIPECAT_MODEL_VERSION}.onnx"

WAVEKAT_ZH_MODEL_URL = (
    "https://huggingface.co/wavekat/smart-turn-ONNX/resolve/main/zh/smart-turn-cpu.onnx"
)
WAVEKAT_ZH_MODEL_VERSION = "wavekat-zh-cpu"
WAVEKAT_ZH_MODEL_CACHE = SCRIPTS / f"smart-turn-{WAVEKAT_ZH_MODEL_VERSION}.onnx"

SAMPLE_RATE = 16_000
BUFFER_SAMPLES = 128_000  # 8 seconds at 16 kHz (matches Rust ring buffer)

# (backend, clip) — drives both the Python pipeline and the entry list written
# to reference.json. Add new rows here, then re-run the script.
TASKS: list[tuple[str, str]] = [
    # Pipecat upstream on English fixtures.
    ("pipecat", "silence_2s.wav"),
    ("pipecat", "speech_finished.wav"),
    ("pipecat", "speech_mid.wav"),
    # Pipecat upstream on Mandarin fixtures (cross-lingual baseline — useful to
    # compare against the wavekat-zh fine-tune below).
    ("pipecat", "zh_speech_finished.wav"),
    ("pipecat", "zh_speech_finished_short.wav"),
    ("pipecat", "zh_speech_mid.wav"),
    # WaveKat zh fine-tune on Mandarin fixtures.
    ("wavekat-zh", "zh_speech_finished.wav"),
    ("wavekat-zh", "zh_speech_finished_short.wav"),
    ("wavekat-zh", "zh_speech_mid.wav"),
]

BACKEND_MODELS: dict[str, tuple[str, Path]] = {
    "pipecat": (PIPECAT_MODEL_URL, PIPECAT_MODEL_CACHE),
    "wavekat-zh": (WAVEKAT_ZH_MODEL_URL, WAVEKAT_ZH_MODEL_CACHE),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_model(url: str, cache: Path) -> Path:
    if cache.exists():
        return cache
    print(f"Downloading model from {url} ...", flush=True)
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, cache)
    print(f"Saved to {cache}", flush=True)
    return cache


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


def infer(audio: np.ndarray, session, extractor) -> tuple[float, np.ndarray]:
    """Run a Smart Turn pipeline on audio.

    All Smart Turn variants share the same feature extractor and tensor I/O,
    so a single helper works for both pipecat and wavekat models.

    Returns:
        (probability, mel_tensor) where mel_tensor has shape [80, 800].
    """
    features = extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="np")
    input_features = features["input_features"].astype(np.float32)  # [1, 80, 800]
    outputs = session.run(None, {"input_features": input_features})
    probability = float(np.squeeze(outputs[0]))  # already a sigmoid probability in [0, 1]
    mel = input_features[0]  # [80, 800]
    return probability, mel


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

    extractor = WhisperFeatureExtractor(chunk_length=8)

    # One ORT session per backend, reused across that backend's clips.
    sessions: dict[str, "ort.InferenceSession"] = {}
    for backend, (url, cache) in BACKEND_MODELS.items():
        if any(b == backend for b, _ in TASKS):
            model_path = ensure_model(url, cache)
            sessions[backend] = ort.InferenceSession(str(model_path))

    results = []
    for backend, name in TASKS:
        path = FIXTURES / name
        if not path.exists():
            print(f"ERROR: missing fixture {path}", file=sys.stderr)
            sys.exit(1)
        audio = load_audio(path)
        prob, mel = infer(audio, sessions[backend], extractor)
        # Only save mel fixtures for the pipecat backend; the mel preprocessing
        # is identical across Smart Turn variants, so one set is enough.
        if backend == "pipecat":
            np.save(str(FIXTURES / f"{name}.mel.npy"), mel)
        print(f"  [{backend}] {name}: probability = {prob:.4f}")
        results.append(
            {
                "backend": backend,
                "file": name,
                "probability": round(prob, 6),
            }
        )

    out_path = FIXTURES / "reference.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

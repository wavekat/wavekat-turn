# Data Structures

Artifacts produced by the notebook pipeline (`01-asr-transcribe`, `02-vad`) and their schemas.

## Directory Layout

```
data/
├── wav/                          # Source audio
│   ├── R8001_M8004_MS801.wav     # 8-ch, 16 kHz, PCM-16
│   └── R8003_M8001_MS801.wav
├── asr_results/                  # Per-file ASR transcriptions
│   ├── R8001_M8004_MS801.json
│   └── R8003_M8001_MS801.json
└── vad_probs/                    # Per-frame speech probabilities
    ├── R8001_M8004_MS801.npy
    └── R8003_M8001_MS801.npy
```

## Source Audio (`data/wav/*.wav`)

| Property    | Value                          |
|-------------|--------------------------------|
| Format      | RIFF WAVE, PCM 16-bit          |
| Channels    | 8 (per-speaker headset mics)   |
| Sample rate | 16 kHz                         |
| Source      | AliMeeting (SLR-119) meetings  |

## ASR Results (`data/asr_results/*.json`)

**Format**: JSON — one file per WAV, named `{wav_stem}.json`.
**Producer**: `notebooks/01-asr-transcribe.ipynb` (Paraformer-zh + FSMN-VAD + ct-punc).

### File Schema

Each file contains a JSON array of record objects:

```jsonc
[
  {
    "text": "全文转写结果...",            // full transcription (punctuated)
    "sentences": [ /* see below */ ],
    "timestamp": [ /* see below */ ]
  }
]
```

### `sentences` Array Element

Each element is one sentence/chunk segmented by the ASR model.

```jsonc
{
  "text":      "啊，",         // punctuated text
  "raw_text":  "啊",           // text without punctuation
  "start":     7130,           // start time (ms)
  "end":       7370,           // end time (ms)
  "timestamp": [[7130, 7370]]  // per-word [start_ms, end_ms] pairs
}
```

| Field       | Type             | Unit | Description                              |
|-------------|------------------|------|------------------------------------------|
| `text`      | string           | —    | Sentence with restored punctuation       |
| `raw_text`  | string           | —    | Same sentence, no punctuation            |
| `start`     | int              | ms   | Sentence start time                      |
| `end`       | int              | ms   | Sentence end time                        |
| `timestamp` | array of [int, int] | ms | Per-word start/end pairs (10 ms frames) |

### Top-level `timestamp` Array

Flat array of all per-word `[start_ms, end_ms]` pairs across the entire file (same data as the union of per-sentence timestamps).

## VAD Probabilities (`data/vad_probs/*.npy`)

**Format**: NumPy `.npy`, 1-D `float32` array.
**Producer**: `notebooks/02-vad.ipynb` (Silero VAD).

| Property          | Value                         |
|-------------------|-------------------------------|
| Shape             | `(num_frames,)`               |
| Dtype             | `float32`                     |
| Frame size        | 512 samples = **32 ms** @ 16 kHz |
| Value range       | `[0.0, 1.0]` — P(speech)     |
| Index → time      | `frame[i]` → `i * 32 ms`     |

### Loading

```python
import numpy as np
probs = np.load("data/vad_probs/R8001_M8004_MS801.npy")
# probs[i] = speech probability at time i * 32 ms
```

### File Details

| File                      | Frames  | Duration   |
|---------------------------|---------|------------|
| `R8001_M8004_MS801.npy`   | 49,183  | ~1,573.9 s |
| `R8003_M8001_MS801.npy`   | 64,625  | ~2,068.0 s |

## Cross-referencing ASR and VAD

ASR timestamps are in **milliseconds**; VAD frames are **32 ms** each.

```python
# Convert ASR ms timestamp to VAD frame index
vad_frame = asr_start_ms // 32

# Convert VAD frame index to ms
time_ms = vad_frame * 32
```

This alignment is used in the next pipeline step (filler candidate extraction): regions where VAD is active (`prob > threshold`) but ASR produces no recognized words.

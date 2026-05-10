# VAD Model Comparison for Chinese Filler Detection

## Use Case

Detect speech regions in Chinese podcast audio, including soft fillers
(呃/嗯/啊) that ASR typically skips. The goal: **VAD active ∧ ASR silent =
filler candidates** (PodcastFillers, Zhu et al. 2022).

Key requirements:
- Catches soft/quiet speech (fillers are often quiet)
- Configurable activation threshold (paper found 0.1 critical)
- Fine temporal resolution (10ms ideal)
- Runs on MacBook Pro (Apple Silicon)

## Comparison

| | FSMN-VAD | Silero VAD | WebRTC VAD | pyannote.audio | TEN VAD |
|---|---|---|---|---|---|
| **Source** | Alibaba / FunASR | Silero | Google | pyannote | TEN Framework |
| **Params** | 0.4M | ~70KB (quantized) | <1MB compiled | ~68M | lightweight ONNX |
| **Resolution** | 200ms chunks | 10ms+ configurable | 10/20/30ms fixed | 10ms | 10–16ms |
| **Threshold** | Configurable (speech_noise_thres) | Probability output, fully tunable | Aggressiveness 0–3 (coarse) | Probability output, fully tunable | Configurable, default 0.5 |
| **Chinese** | Native (5000h Mandarin) | General (6000+ langs, no zh-specific) | Language-agnostic | Trained on AISHELL + AliMeeting | General |
| **Soft fillers** | Good, but 200ms chunks may blur boundaries | Catches at low threshold (~0.1–0.3) | May miss quiet fillers | Best accuracy on soft boundaries | Less documented |
| **Mac perf** | CPU, fast for 0.4M | ~40μs/chunk on M2 Max | Very fast CPU | Slow on CPU (68M params) | arm64 native + ONNX |
| **MPS/GPU** | Yes (via FunASR) | MLX native | N/A (CPU only) | MPS supported | ONNX only |
| **License** | Model-specific (check HF) | MIT | BSD | MIT | Apache 2.0 |

## Analysis

### FSMN-VAD
- **Pro**: Already in our stack (FunASR). Chinese-native, production-proven.
- **Con**: 200ms chunk size is coarser than ideal. Fine boundary detection
  for short fillers (150–400ms) may lose precision.

### Silero VAD
- **Pro**: Tiny, fast, 10ms resolution, MIT license. Threshold tunable to
  ~0.1 for soft speech. MLX native on Apple Silicon.
- **Con**: Not Chinese-optimized. Lightweight design means less sophisticated
  boundary detection.

### WebRTC VAD
- **Pro**: Mature, fast, 10ms native resolution.
- **Con**: No probability output — binary decisions with coarse aggressiveness
  levels (0–3). Hard to tune for soft fillers. No fine-grained threshold.

### pyannote.audio (segmentation-3.0)
- **Pro**: Best accuracy. 10ms resolution. Trained on Chinese datasets
  (AISHELL, AliMeeting). Best at catching soft speech boundaries.
- **Con**: 68M params — slow on CPU. Overkill if we only need binary VAD.

### TEN VAD
- **Pro**: 10ms resolution, superior precision vs WebRTC and Silero. Apache 2.0.
- **Con**: Newer (2024–2025), fewer production deployments. Less documented
  for Chinese/soft speech.

## Recommendation

**Silero VAD** as primary choice:
- 10ms resolution matches what the paper uses
- Threshold tunable to 0.1 (critical finding from the paper)
- Tiny and fast on MacBook
- MIT license
- Good enough for candidate generation — the classifier stage handles precision

**FSMN-VAD** as comparison baseline since it's already in our pipeline.

If accuracy on soft boundaries proves insufficient, upgrade to **pyannote.audio**.

## References

- [PodcastFillers paper](../refs/2203.pdf) — Section 2.2: VAD threshold 0.1, candidates 150ms–2s
- [Silero VAD](https://github.com/snakers4/silero-vad)
- [FSMN-VAD](https://huggingface.co/funasr/fsmn-vad)
- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [TEN VAD](https://github.com/TEN-framework/ten-vad)

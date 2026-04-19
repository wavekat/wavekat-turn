# Notebooks — Filler Detection Pipeline

Following the PodcastFillers approach (Zhu et al., 2022):
filler candidates = regions where **VAD is active** but **ASR produces no words**.

## Pipeline

```
WAV files
  → 01 ASR transcription  (Paraformer-zh, word-level timestamps)
  → 02 VAD detection       (FSMN-VAD, speech region timestamps)
  → 03 Filler candidates   (VAD ∧ ¬ASR, duration filter 150ms–2s)
  → 04 ...                 (labeling / classification TBD)
```

## Notebooks

| # | Notebook | Input | Output | Description |
|---|----------|-------|--------|-------------|
| 01 | `01-asr-transcribe` | `data/wav/*.wav` | `data/asr_results.jsonl` | Paraformer-zh ASR with word timestamps |
| 02 | `02-vad` | `data/wav/*.wav` | `data/vad_results.jsonl` | FSMN-VAD speech region detection |
| 03 | TBD | ASR + VAD results | `data/filler_candidates.jsonl` | Compute VAD ∧ ¬ASR gaps |

## Reference

- `refs/2203.pdf` — PodcastFillers: Filler Word Detection and Classification (Zhu et al., 2022)

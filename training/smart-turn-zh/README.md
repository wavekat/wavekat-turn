# Smart Turn — Mandarin (smart-turn-zh)

Mandarin-Chinese variant of the Smart Turn detector: fine-tuning the upstream
[Pipecat Smart Turn](../pipecat-smart-turn/) architecture (Whisper-Tiny encoder
+ binary classification head) on Chinese conversational audio.

## Layout

- [`plan-data.md`](plan-data.md) — dataset construction plan (under revision)
- [`research/`](research/) — surveys and open questions feeding the plans
  - [`01-datasets.md`](research/01-datasets.md) — OpenSLR + HuggingFace dataset survey
- `data/` — data pipeline scripts
- `notebooks/` — Jupyter exploration
- `train/` — training scripts (later)

## Status

Design phase. The data pipeline plan in `plan-data.md` is the original
LLM-rewriting + TTS approach; we are revising toward real conversational
corpora — see `research/datasets.md` for the source-corpus analysis.

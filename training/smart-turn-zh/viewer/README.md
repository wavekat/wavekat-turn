# Audio Viewer

Browser-based tool for reviewing audio alongside VAD probabilities and ASR
transcriptions. Built to support the Smart Turn training data pipeline — lets
you visually inspect and cross-reference the outputs of the ASR and VAD
notebooks before feeding data into downstream labelling and training steps.

## What it does

- **Waveform display** with LOD decimation — handles 1-hour, multi-channel
  recordings smoothly. Switch between individual channels or a merged "All" view.
- **VAD probability curve** with dual-threshold hysteresis (entry/exit),
  color-coded active/inactive speech regions.
- **ASR transcript panel** with timestamped sentence list. Click any sentence
  to jump to that moment.
- **Text search** across ASR results with timeline highlighting and
  Previous/Next navigation.
- **Audio playback** with click-to-seek, keyboard shortcuts, and auto-scroll.
- **Minimap** overview bar for quick navigation across long recordings.

All processing is client-side — drop your files in and go, no server needed.

## Quick start

```bash
npm install
npm run dev
```

Then open the URL shown in the terminal. Drop in your files:

| File type | Example | Source |
|-----------|---------|--------|
| `.wav`    | `R8001_M8004_MS801.wav` | `data/wav/` |
| `.npy`    | `R8001_M8004_MS801.npy` | `data/vad_probs/` (from `02-vad.ipynb`) |
| `.json`   | `R8001_M8004_MS801.json` | `data/asr_results/` (from `01-asr-transcribe.ipynb`) |

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| F | Focus search box |
| Enter / Shift+Enter | Next / Previous search result |
| +/- | Zoom in / out |
| Left / Right | Pan |
| 0 | Merged (all channels) view |
| 1-9 | Select channel |

## Further reading

- [`docs/02-data-structures.md`](../docs/02-data-structures.md) — schemas for
  the WAV, ASR, and VAD data formats this viewer consumes.
- [`docs/03-audio-viewer.md`](../docs/03-audio-viewer.md) — full design plan,
  architecture decisions, and performance strategy.
- [`notebooks/`](../notebooks/) — the pipeline notebooks that produce the data.

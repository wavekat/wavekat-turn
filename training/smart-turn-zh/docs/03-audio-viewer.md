# Audio Viewer — Plan

A browser-based tool for inspecting audio alongside VAD probabilities and ASR
transcriptions. Designed for 1-hour recordings at interactive frame rates.

## Goal

Accelerate data review for the training pipeline. Load a WAV file with its
matching VAD and ASR outputs, visualize everything on a synchronized timeline,
and search/navigate ASR text to jump to specific moments.

## Requirements

### Functional

1. **File upload** — drag-and-drop or file picker for three inputs:
   - `.wav` audio (any channel count, 16 kHz, PCM-16)
   - `.npy` VAD probabilities (float32, 32 ms frames)
   - `.json` ASR result (sentence-level timestamps)

2. **Waveform display** — amplitude over time. Supports any channel count
   (1, 2, 4, 8, ...). Individual channel selection or a merged "all channels"
   view that overlays every channel on a single waveform.

3. **Spectrogram display** — time-frequency view of the selected channel.

4. **VAD probability curve** — stacked below the waveform, showing P(speech)
   per 32 ms frame. Two configurable threshold lines: **entry** (e.g. 0.3) and
   **exit** (e.g. 0.1). Hysteresis logic: speech region starts when probs rise
   above entry, ends when they drop below exit. Active speech regions are
   filled with one color; inactive regions with another.

5. **ASR transcript panel** — scrollable sentence list with timestamps. Clicking
   a sentence seeks the playback cursor and centers the viewport.

6. **Text search** — search box filters ASR sentences by keyword. Highlights
   matching segments on the timeline. Previous/Next buttons (+ keyboard
   shortcuts) cycle through results.

7. **Audio playback** — play/pause, click-to-seek on waveform, playback cursor
   that scrolls the viewport.

### Non-functional

- Handle **1-hour audio** (57.6 M samples per channel) without lag.
- Smooth zoom from full-file overview down to individual samples.
- All processing client-side — no server required. Open `index.html` and go.

## Data Inputs

All three formats are documented in `02-data-structures.md`. Key details
relevant to the viewer:

| Input | Size (1 hr) | Notes |
|-------|-------------|-------|
| WAV (N-ch, 16 kHz, 16-bit) | varies | Decode via Web Audio API; select channel or merge all |
| VAD .npy (float32) | ~450 KB | ~112 K frames; parse npy header + raw ArrayBuffer |
| ASR .json | ~50-200 KB | Array of records with `sentences[].{start,end,text}` |

### Parsing `.npy` in the browser

NumPy v1.0 `.npy` format: 6-byte magic + 2-byte version + 2-byte header length
+ ASCII header (dtype, shape, order) + raw data. For 1-D float32 arrays this is
trivial — read the header to get length, then wrap the remaining bytes as a
`Float32Array`.

## Architecture

```
viewer/
├── index.html          # single entry point
├── src/
│   ├── main.ts         # bootstrap, file loading, layout
│   ├── audio.ts        # WAV decode, channel extraction, LOD cache
│   ├── waveform.ts     # Canvas waveform renderer
│   ├── spectrogram.ts  # Canvas spectrogram renderer (FFT in Worker)
│   ├── vad.ts          # .npy parser + VAD curve renderer
│   ├── asr.ts          # ASR JSON loader + transcript panel + search
│   ├── timeline.ts     # shared time axis, zoom/pan state, cursor sync
│   └── fft-worker.ts   # Web Worker for spectrogram computation
├── package.json
├── tsconfig.json
└── vite.config.ts
```

**Stack**: TypeScript + Vite (dev server + build). No framework — vanilla DOM
for the panels, Canvas 2D for all visualizations. Keeps the bundle tiny and
avoids framework overhead on large data.

## Performance Strategy

### Waveform — level-of-detail (LOD) decimation

Raw samples per channel for 1 hr: **57.6 M**. A screen is ~2000 px wide.
Drawing all samples is pointless and slow.

**Approach**: pre-compute a min/max mipmap pyramid on load.

```
Level 0:  raw samples           (57.6 M points)
Level 1:  min/max per 64 samples  (~900 K points)
Level 2:  min/max per 256         (~225 K points)
Level 3:  min/max per 1024        (~56 K points)
Level 4:  min/max per 4096        (~14 K points)
...
```

At render time, pick the level where each pixel covers ~1 bucket. Draw a filled
shape between min and max per pixel column. This is O(screen_width) per frame
regardless of zoom level.

Build the pyramid in a **Web Worker** so the UI stays responsive during load.

### Spectrogram — viewport-only FFT

Computing a full STFT for 1 hour is expensive (~3.6 M frames at hop=16).
Instead:

1. Compute FFT **only for the visible time range** plus a small margin.
2. Cache computed tiles (e.g. 10-second chunks) in a tile map.
3. On pan/zoom, render cached tiles and compute missing ones in a Worker.
4. FFT size: 512 (32 ms at 16 kHz) — matches VAD frame size, gives 256 freq
   bins up to 8 kHz. Fast enough for real-time viewport updates.

Render to an off-screen canvas, then `drawImage` to the visible canvas —
avoids per-pixel DOM work.

### VAD curve

~112 K points for 1 hour. Apply the same LOD approach as waveform (average
instead of min/max). At full zoom-out the curve is already only ~2 K points
per screen width — nearly free.

### ASR overlay

Sentence count is small (a few hundred). No special optimization needed.
On search, build a Set of matching sentence indices, then highlight their
time spans on the timeline canvas.

### General

- **Typed arrays everywhere** — `Float32Array` / `Int16Array` for audio and
  VAD data. No JS array copies.
- **`requestAnimationFrame`** for synchronized redraws — batch all dirty
  canvases into one frame.
- **Debounced zoom/pan** — pointer events update viewport state; rendering
  reads state on rAF.
- **Channel switching** — re-slice from the decoded `AudioBuffer` (kept in
  memory), rebuild LOD pyramid. ~1 s for 1-hour file. "All" mode averages
  channels into a single waveform, or overlays them with per-channel colors.

## UI Layout

```
┌──────────────────────────────────────────────────────┐
│  [Drop zone / file pickers]  [Channel ▾ All]  [▶ ⏸]  │
├──────────────────────────────────────────────────────┤
│  Waveform                  ░░░▓▓▓▓▓▓░░░░▓▓▓▓░░░░░░  │  ← Canvas
│  VAD probs        ─────╱╲──╱╲╲──────╱╲───────────── │  ← Canvas (stacked)
│  Spectrogram      ▒▒▒▓▓▓▓▒▒▒▒▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  │  ← Canvas
├────────────────────────────┬─────────────────────────┤
│  [🔍 Search ASR...]       │  Time axis / minimap     │
│  ─────────────────────────│──────────────────────────│
│  00:01.23  啊，我觉得...  │                          │
│  00:05.67  对对对，就是   │                          │
│  00:12.34  然后我们...    │  (overview bar showing   │
│  ...                      │   full file with viewport │
│  [◀ Prev] [Next ▶]       │   indicator)              │
└────────────────────────────┴─────────────────────────┘
```

- **Top row**: file upload area, channel selector, playback controls.
- **Middle**: three vertically stacked canvases sharing a time axis. Scroll
  wheel zooms, click-drag pans. All canvases pan/zoom in sync.
- **Bottom left**: ASR transcript panel with search. Clicking a sentence
  scrolls the timeline. Search highlights appear on the waveform/VAD canvases.
- **Bottom right**: minimap showing the full file duration. The blue rectangle
  indicates the current viewport; drag to navigate.

### Keyboard shortcuts

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| F | Search (focus search box) |
| Enter / Shift+Enter | Next / Previous search result |
| +/- | Zoom in / out |
| Left / Right | Pan |
| 0 | Merged (all channels) view |
| 1-9 | Select channel |

## Implementation Phases

### Phase 1 — Scaffold + waveform

- Vite project setup, HTML shell, file drop zone.
- WAV decode via `AudioContext.decodeAudioData`.
- LOD pyramid builder (Web Worker).
- Canvas waveform renderer with pan/zoom.
- Channel selector (individual + "All" merged view).
- Audio playback with seek.

### Phase 2 — VAD + ASR

- `.npy` parser (typed array).
- VAD probability canvas, synced to waveform timeline.
- Threshold line (draggable or input).
- ASR JSON loader, transcript panel.
- Click sentence to seek.

### Phase 3 — Search + polish

- ASR text search with highlight on timeline.
- Prev/Next navigation.
- Minimap overview bar.
- Keyboard shortcuts.
- Responsive layout, loading indicators, error handling.

### Phase 4 — Spectrogram

- FFT Web Worker with tile caching.
- Spectrogram canvas, synced timeline.
- Color map (e.g. viridis or magma).

import type { Timeline } from './timeline';
import type { AudioStore } from './audio';

/* ---- Mel helpers ---- */

function hzToMel(hz: number): number {
  return 2595 * Math.log10(1 + hz / 700);
}

function melToHz(mel: number): number {
  return 700 * (10 ** (mel / 2595) - 1);
}

/* ---- Turbo colormap (256 entries) ---- */

const STOPS: [number, number, number, number][] = [
  [0,     0,   0,  10],
  [0.1,  20,   10, 120],
  [0.2,  30,   80, 200],
  [0.3,   0,  160, 230],
  [0.4,   0,  210, 170],
  [0.5,  50,  220,  80],
  [0.6, 140,  210,  20],
  [0.7, 210,  180,   0],
  [0.8, 250,  130,   0],
  [0.9, 240,   50,   0],
  [1,   180,   10,   0],
];

const CMAP = new Uint8Array(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  let si = 0;
  for (let s = 1; s < STOPS.length; s++) {
    if (STOPS[s][0] >= t) {
      si = s - 1;
      break;
    }
  }
  const [t0, r0, g0, b0] = STOPS[si];
  const [t1, r1, g1, b1] = STOPS[si + 1];
  const f = (t - t0) / (t1 - t0);
  CMAP[i * 3] = Math.round(r0 + (r1 - r0) * f);
  CMAP[i * 3 + 1] = Math.round(g0 + (g1 - g0) * f);
  CMAP[i * 3 + 2] = Math.round(b0 + (b1 - b0) * f);
}

/* ---- Renderer ---- */

export type FreqScale = 'mel' | 'linear';

interface CachedTile {
  magnitudes: Float32Array;
  frames: number;
  freqBins: number;
  canvas: HTMLCanvasElement;
  scale: FreqScale;
  startTime: number;
  endTime: number;
}

export class SpectrogramRenderer {
  channel = -1;
  freqScale: FreqScale = 'mel';
  dbMin = -80;
  dbMax = 0;

  /** Called after a tile finishes computing so the component can redraw. */
  onTileReady: (() => void) | null = null;

  private ctx: CanvasRenderingContext2D;
  private worker: Worker;
  private cache = new Map<string, CachedTile>();
  private pending = new Set<string>();
  private lastRaw: Int16Array | null = null;

  private readonly fftSize = 512;
  private readonly hopSize = 256;
  private readonly tileSec = 10;
  private readonly maxPending = 20;

  constructor(
    private canvas: HTMLCanvasElement,
    private tl: Timeline,
    private audio: AudioStore,
  ) {
    this.ctx = canvas.getContext('2d')!;
    this.worker = new Worker(
      new URL('./fft-worker.ts', import.meta.url),
      { type: 'module' },
    );
    this.worker.onmessage = (e) => this.onWorkerMsg(e);
  }

  resize() {
    const dpr = devicePixelRatio;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
  }

  render() {
    if (!this.audio.raw) return;

    // Invalidate cache when audio data changes (new file loaded)
    if (this.audio.raw !== this.lastRaw) {
      this.clearCache();
      this.lastRaw = this.audio.raw;
    }

    const { ctx, canvas, tl } = this;
    const dpr = devicePixelRatio;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Visible tile range + 1-tile margin on each side
    const maxTile = Math.ceil(this.audio.duration / this.tileSec);
    const t0 = Math.max(0, Math.floor(tl.viewStart / this.tileSec) - 1);
    const t1 = Math.min(maxTile, Math.ceil(tl.viewEnd / this.tileSec) + 1);

    for (let ti = t0; ti < t1; ti++) {
      const key = `${this.channel}:${ti}`;
      const cached = this.cache.get(key);

      if (cached) {
        // Rebuild tile canvas if freq scale changed
        if (cached.scale !== this.freqScale) {
          this.rebuildTileCanvas(cached);
        }

        const x0 = tl.timeToX(cached.startTime, w);
        const x1 = tl.timeToX(cached.endTime, w);
        const left = Math.max(0, x0);
        const right = Math.min(w, x1);
        if (right > left) {
          const srcX = ((left - x0) / (x1 - x0)) * cached.canvas.width;
          const srcW = ((right - left) / (x1 - x0)) * cached.canvas.width;
          ctx.drawImage(
            cached.canvas,
            srcX, 0, srcW, cached.canvas.height,
            left, 0, right - left, h,
          );
        }
      } else if (!this.pending.has(key) && this.pending.size < this.maxPending) {
        this.requestTile(ti);
      }
    }

    this.drawFreqAxis(w, h);

    // Cursor
    const cx = tl.timeToX(tl.cursor, w);
    if (cx >= 0 && cx <= w) {
      ctx.strokeStyle = '#ff5722';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(cx, 0);
      ctx.lineTo(cx, h);
      ctx.stroke();
    }
  }

  dispose() {
    this.worker.terminate();
    this.cache.clear();
  }

  /* ---- internals ---- */

  private clearCache() {
    this.cache.clear();
    this.pending.clear();
  }

  private requestTile(ti: number) {
    const key = `${this.channel}:${ti}`;
    this.pending.add(key);

    const sr = this.audio.sampleRate;
    const s0 = ti * this.tileSec * sr;
    const s1 = Math.min((ti + 1) * this.tileSec * sr, this.audio.totalFrames);
    const samples = this.extractSamples(s0, s1);

    this.worker.postMessage(
      {
        type: 'compute',
        tileKey: key,
        samples,
        fftSize: this.fftSize,
        hopSize: this.hopSize,
      },
      { transfer: [samples.buffer] },
    );
  }

  private extractSamples(start: number, end: number): Float32Array {
    const raw = this.audio.raw!;
    const nch = this.audio.channelCount;
    const ch = this.channel;
    const len = end - start;
    const out = new Float32Array(len);

    if (ch === -1) {
      for (let i = 0; i < len; i++) {
        let sum = 0;
        for (let c = 0; c < nch; c++) sum += raw[(start + i) * nch + c];
        out[i] = sum / nch / 32768;
      }
    } else {
      for (let i = 0; i < len; i++) {
        out[i] = raw[(start + i) * nch + ch] / 32768;
      }
    }
    return out;
  }

  private onWorkerMsg(e: MessageEvent) {
    const { tileKey, magnitudes, frames, freqBins } = e.data as {
      tileKey: string;
      magnitudes: Float32Array;
      frames: number;
      freqBins: number;
    };
    this.pending.delete(tileKey);
    if (frames === 0) return;

    const tileCanvas = this.buildTileCanvas(magnitudes, frames, freqBins);

    const ti = parseInt(tileKey.split(':')[1]);
    const startTime = ti * this.tileSec;
    const endTime = Math.min(startTime + this.tileSec, this.audio.duration);

    this.cache.set(tileKey, {
      magnitudes, frames, freqBins,
      canvas: tileCanvas,
      scale: this.freqScale,
      startTime, endTime,
    });
    this.onTileReady?.();
  }

  /** Build an offscreen canvas from magnitude data using current freqScale. */
  private buildTileCanvas(
    magnitudes: Float32Array, frames: number, freqBins: number,
  ): HTMLCanvasElement {
    const img = this.buildImageData(magnitudes, frames, freqBins, this.freqScale);
    const oc = document.createElement('canvas');
    oc.width = frames;
    oc.height = freqBins;
    oc.getContext('2d')!.putImageData(img, 0, 0);
    return oc;
  }

  /** Rebuild an existing tile's canvas when freq scale changes. */
  private rebuildTileCanvas(tile: CachedTile) {
    const img = this.buildImageData(
      tile.magnitudes, tile.frames, tile.freqBins, this.freqScale,
    );
    tile.canvas.getContext('2d')!.putImageData(img, 0, 0);
    tile.scale = this.freqScale;
  }

  /** Convert magnitude grid to RGBA ImageData with mel or linear freq mapping. */
  private buildImageData(
    magnitudes: Float32Array, frames: number, freqBins: number, scale: FreqScale,
  ): ImageData {
    const img = new ImageData(frames, freqBins);
    const px = img.data;
    const range = this.dbMax - this.dbMin;
    const nyquist = this.audio.sampleRate / 2;
    const melMax = scale === 'mel' ? hzToMel(nyquist) : 0;

    for (let row = 0; row < freqBins; row++) {
      // Map output row to a (possibly fractional) FFT bin index
      let k: number;
      if (scale === 'mel') {
        const mel = (row / (freqBins - 1)) * melMax;
        const hz = melToHz(mel);
        k = (hz / nyquist) * (freqBins - 1);
      } else {
        k = row;
      }

      const k0 = Math.floor(k);
      const k1 = Math.min(k0 + 1, freqBins - 1);
      const frac = k - k0;
      const y = freqBins - 1 - row; // flip: low freq at bottom

      for (let f = 0; f < frames; f++) {
        const base = f * freqBins;
        const db = magnitudes[base + k0] * (1 - frac) + magnitudes[base + k1] * frac;
        const norm = Math.max(0, Math.min(1, (db - this.dbMin) / range));
        const ci = Math.round(norm * 255) * 3;
        const pi = (y * frames + f) * 4;
        px[pi] = CMAP[ci];
        px[pi + 1] = CMAP[ci + 1];
        px[pi + 2] = CMAP[ci + 2];
        px[pi + 3] = 255;
      }
    }

    return img;
  }

  private drawFreqAxis(w: number, h: number) {
    const { ctx } = this;
    const nyquist = this.audio.sampleRate / 2;
    const freqs = [500, 1000, 2000, 4000, 6000, 8000].filter((f) => f <= nyquist);

    ctx.font = '9px monospace';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';

    for (const freq of freqs) {
      let y: number;
      if (this.freqScale === 'mel') {
        y = h * (1 - hzToMel(freq) / hzToMel(nyquist));
      } else {
        y = h * (1 - freq / nyquist);
      }

      ctx.strokeStyle = 'rgba(255,255,255,0.1)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();

      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      const label = freq >= 1000 ? `${freq / 1000}k` : `${freq}`;
      ctx.fillText(label, w - 4, y);
    }
  }
}

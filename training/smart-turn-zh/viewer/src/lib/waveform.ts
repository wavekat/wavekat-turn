import type { Timeline } from './timeline';
import type { AudioStore } from './audio';

export type WaveformScale = 'linear' | 'dB';

export class WaveformRenderer {
  channel = -1;
  scale: WaveformScale = 'dB';
  private ctx: CanvasRenderingContext2D;

  constructor(
    private canvas: HTMLCanvasElement,
    private tl: Timeline,
    private audio: AudioStore,
  ) {
    this.ctx = canvas.getContext('2d')!;
  }

  resize() {
    const dpr = devicePixelRatio;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
  }

  render() {
    if (!this.audio.raw) return;
    const { ctx, canvas, tl, audio } = this;
    const dpr = devicePixelRatio;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = '#333';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    if (this.scale === 'dB') {
      const mid = h / 2, amp = mid * 0.95;
      ctx.font = '9px monospace';
      ctx.textBaseline = 'middle';
      for (const db of [-6, -12, -24, -48]) {
        const norm = Math.max(0, (db + 60) / 60);
        const yUp = mid - norm * amp;
        const yDn = mid + norm * amp;
        ctx.strokeStyle = 'rgba(255,255,255,0.08)';
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.moveTo(0, yUp); ctx.lineTo(w, yUp);
        ctx.moveTo(0, yDn); ctx.lineTo(w, yDn);
        ctx.stroke();
        ctx.fillStyle = 'rgba(255,255,255,0.2)';
        ctx.fillText(`${db}`, 2, yUp);
      }
    }

    const sr = audio.sampleRate;
    const s0 = Math.floor(tl.viewStart * sr);
    const s1 = Math.ceil(tl.viewEnd * sr);
    const spp = (s1 - s0) / w;

    if (spp <= 1) {
      this.drawRaw(w, h, s0, s1);
    } else if (spp <= 512) {
      this.drawDirect(w, h, s0, s1);
    } else {
      this.drawLOD(w, h, s0, s1, spp);
    }

    // Loop range highlight
    if (tl.hasLoop) {
      const x0 = Math.max(0, tl.timeToX(tl.loopStart, w));
      const x1 = Math.min(w, tl.timeToX(tl.loopEnd, w));
      if (x1 > x0) {
        ctx.fillStyle = 'rgba(255, 152, 0, 0.15)';
        ctx.fillRect(x0, 0, x1 - x0, h);
        ctx.strokeStyle = 'rgba(255, 152, 0, 0.6)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x0, 0); ctx.lineTo(x0, h);
        ctx.moveTo(x1, 0); ctx.lineTo(x1, h);
        ctx.stroke();
      }
    }

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

  private mapY(v: number, mid: number, amp: number): number {
    if (this.scale === 'dB') {
      const sign = v < 0 ? -1 : 1;
      const abs = Math.abs(v);
      const db = abs > 1e-6 ? 20 * Math.log10(abs) : -60;
      const norm = Math.max(0, (db + 60) / 60);
      return mid - sign * norm * amp;
    }
    return mid - v * amp;
  }

  private drawRaw(w: number, h: number, s0: number, s1: number) {
    const { ctx, audio } = this;
    const mid = h / 2, amp = mid * 0.95;
    ctx.strokeStyle = '#4fc3f7';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < w; x++) {
      const idx = s0 + Math.round((x / w) * (s1 - s0));
      const y = this.mapY(audio.sample(idx, this.channel), mid, amp);
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  private drawDirect(w: number, h: number, s0: number, s1: number) {
    const { ctx, audio } = this;
    const mid = h / 2, amp = mid * 0.95;
    const span = s1 - s0;

    ctx.fillStyle = '#4fc3f7';
    ctx.beginPath();

    const mins = new Float32Array(w);
    const maxs = new Float32Array(w);
    for (let x = 0; x < w; x++) {
      const start = s0 + Math.floor((x / w) * span);
      const end = s0 + Math.floor(((x + 1) / w) * span);
      let lo = Infinity, hi = -Infinity;
      for (let i = start; i < end; i++) {
        const v = audio.sample(i, this.channel);
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
      mins[x] = isFinite(lo) ? lo : 0;
      maxs[x] = isFinite(hi) ? hi : 0;
    }

    for (let x = 0; x < w; x++) {
      const y = this.mapY(maxs[x], mid, amp);
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    for (let x = w - 1; x >= 0; x--) {
      ctx.lineTo(x, this.mapY(mins[x], mid, amp));
    }
    ctx.closePath();
    ctx.fill();
  }

  private drawLOD(w: number, h: number, s0: number, s1: number, spp: number) {
    const levels = this.audio.getLOD(this.channel);
    if (!levels.length) return;

    let lv = levels[0];
    for (const l of levels) {
      if (l.bucketSize <= spp) lv = l;
      else break;
    }

    const { ctx } = this;
    const mid = h / 2, amp = mid * 0.95;
    const span = s1 - s0;

    ctx.fillStyle = '#4fc3f7';
    ctx.beginPath();

    for (let x = 0; x < w; x++) {
      const ss = s0 + (x / w) * span;
      const se = ss + span / w;
      const b0 = Math.max(0, Math.floor(ss / lv.bucketSize));
      const b1 = Math.min(Math.ceil(se / lv.bucketSize), lv.max.length);
      let hi = -Infinity;
      for (let b = b0; b < b1; b++) if (lv.max[b] > hi) hi = lv.max[b];
      if (!isFinite(hi)) hi = 0;
      const y = this.mapY(hi, mid, amp);
      x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    for (let x = w - 1; x >= 0; x--) {
      const ss = s0 + (x / w) * span;
      const se = ss + span / w;
      const b0 = Math.max(0, Math.floor(ss / lv.bucketSize));
      const b1 = Math.min(Math.ceil(se / lv.bucketSize), lv.min.length);
      let lo = Infinity;
      for (let b = b0; b < b1; b++) if (lv.min[b] < lo) lo = lv.min[b];
      if (!isFinite(lo)) lo = 0;
      ctx.lineTo(x, this.mapY(lo, mid, amp));
    }
    ctx.closePath();
    ctx.fill();
  }
}

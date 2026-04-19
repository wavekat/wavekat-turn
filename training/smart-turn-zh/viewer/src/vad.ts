import type { Timeline } from './timeline';

export class VADRenderer {
  probs: Float32Array | null = null;
  entryThreshold = 0.3;
  exitThreshold = 0.1;
  readonly frameSec = 0.032; // 32 ms per frame

  private ctx: CanvasRenderingContext2D;

  constructor(
    private canvas: HTMLCanvasElement,
    private tl: Timeline,
  ) {
    this.ctx = canvas.getContext('2d')!;
  }

  load(buffer: ArrayBuffer) {
    this.probs = parseNpy(buffer);
  }

  resize() {
    const dpr = devicePixelRatio;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
  }

  render() {
    if (!this.probs) return;
    const { ctx, canvas, tl, probs, frameSec } = this;
    const dpr = devicePixelRatio;
    const w = canvas.width / dpr;
    const h = canvas.height / dpr;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const vStart = tl.viewStart;
    const vSpan = tl.viewEnd - vStart;
    const startFrame = Math.max(0, Math.floor(vStart / frameSec) - 1);
    const endFrame = Math.min(probs.length, Math.ceil(tl.viewEnd / frameSec) + 1);

    // Compute hysteresis state up to startFrame
    let active = false;
    for (let i = 0; i < startFrame; i++) {
      if (!active && probs[i] >= this.entryThreshold) active = true;
      else if (active && probs[i] <= this.exitThreshold) active = false;
    }

    // Draw active/inactive regions
    let regStart = startFrame;
    let regActive = active;

    const fillRegion = (from: number, to: number, isActive: boolean) => {
      const x1 = ((from * frameSec - vStart) / vSpan) * w;
      const x2 = ((to * frameSec - vStart) / vSpan) * w;
      const left = Math.max(0, x1), right = Math.min(w, x2);
      if (right <= left) return;
      ctx.fillStyle = isActive ? 'rgba(76,175,80,0.25)' : 'rgba(255,255,255,0.03)';
      ctx.fillRect(left, 0, right - left, h);
    };

    for (let i = startFrame; i < endFrame; i++) {
      const was: boolean = active;
      if (!active && probs[i] >= this.entryThreshold) active = true;
      else if (active && probs[i] <= this.exitThreshold) active = false;
      if (active !== was) {
        fillRegion(regStart, i, regActive);
        regStart = i;
        regActive = active;
      }
    }
    fillRegion(regStart, endFrame, regActive);

    // Probability curve
    ctx.strokeStyle = '#4fc3f7';
    ctx.lineWidth = 1;
    ctx.beginPath();
    let first = true;
    for (let x = 0; x < w; x++) {
      const t = vStart + (x / w) * vSpan;
      const fi = Math.round(t / frameSec);
      if (fi < 0 || fi >= probs.length) continue;
      const y = h - probs[fi] * h;
      if (first) { ctx.moveTo(x, y); first = false; } else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Threshold lines
    this.drawThresh(w, h, this.entryThreshold, '#ff9800', 'Entry');
    this.drawThresh(w, h, this.exitThreshold, '#f44336', 'Exit');

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

  private drawThresh(w: number, h: number, val: number, color: string, label: string) {
    const y = h - val * h;
    const { ctx } = this;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.font = '10px monospace';
    ctx.fillText(`${label} ${val}`, 4, y - 3);
  }
}

function parseNpy(buf: ArrayBuffer): Float32Array {
  const bytes = new Uint8Array(buf);
  const major = bytes[6];
  let headerLen: number, dataOffset: number;
  if (major <= 1) {
    headerLen = new DataView(buf).getUint16(8, true);
    dataOffset = 10 + headerLen;
  } else {
    headerLen = new DataView(buf).getUint32(8, true);
    dataOffset = 12 + headerLen;
  }
  if (dataOffset % 4 !== 0) {
    return new Float32Array(buf.slice(dataOffset));
  }
  return new Float32Array(buf, dataOffset);
}

export class Timeline {
  duration = 0;
  viewStart = 0;
  viewEnd = 1;
  cursor = 0;
  sampleRate = 16000;

  private listeners: Array<() => void> = [];

  setDuration(d: number) {
    this.duration = d;
    this.viewStart = 0;
    this.viewEnd = d;
    this.emit();
  }

  setView(start: number, end: number) {
    const minSpan = 0.005;
    let s = start, e = end;
    if (e - s < minSpan) {
      const mid = (s + e) / 2;
      s = mid - minSpan / 2;
      e = mid + minSpan / 2;
    }
    if (s < 0) { e -= s; s = 0; }
    if (e > this.duration) { s -= e - this.duration; e = this.duration; }
    if (s < 0) s = 0;
    this.viewStart = s;
    this.viewEnd = e;
    this.emit();
  }

  setCursor(t: number) {
    this.cursor = Math.max(0, Math.min(this.duration, t));
    this.emit();
  }

  zoom(factor: number, anchorFrac: number) {
    const span = this.viewEnd - this.viewStart;
    const anchor = this.viewStart + span * anchorFrac;
    const newSpan = span * factor;
    this.setView(anchor - newSpan * anchorFrac, anchor - newSpan * anchorFrac + newSpan);
  }

  pan(deltaSec: number) {
    this.setView(this.viewStart + deltaSec, this.viewEnd + deltaSec);
  }

  timeToX(t: number, width: number): number {
    return ((t - this.viewStart) / (this.viewEnd - this.viewStart)) * width;
  }

  xToTime(x: number, width: number): number {
    return this.viewStart + (x / width) * (this.viewEnd - this.viewStart);
  }

  /** Fire listeners without changing state. Used by the playback loop. */
  flush() {
    this.emit();
  }

  onUpdate(fn: () => void): () => void {
    this.listeners.push(fn);
    return () => {
      const idx = this.listeners.indexOf(fn);
      if (idx >= 0) this.listeners.splice(idx, 1);
    };
  }

  private emit() {
    for (const fn of this.listeners) fn();
  }
}

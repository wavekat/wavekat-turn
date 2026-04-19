import type { Timeline } from './timeline';

export interface Sentence {
  text: string;
  start: number; // ms
  end: number; // ms
}

export class ASRPanel {
  sentences: Sentence[] = [];
  results: number[] = [];
  resultIdx = -1;

  private query = '';
  private listEl: HTMLElement;
  private countEl: HTMLElement;

  constructor(
    container: HTMLElement,
    private tl: Timeline,
  ) {
    this.listEl = container.querySelector('#transcript-list')!;
    this.countEl = container.querySelector('#search-count')!;
  }

  load(json: unknown[]) {
    this.sentences = [];
    for (const rec of json as Array<{ sentences?: Sentence[] }>) {
      if (rec.sentences) {
        for (const s of rec.sentences) {
          this.sentences.push({ text: s.text ?? '', start: s.start, end: s.end });
        }
      }
    }
    this.renderList();
  }

  search(q: string) {
    this.query = q;
    this.results = [];
    this.resultIdx = -1;
    if (q) {
      const lower = q.toLowerCase();
      for (let i = 0; i < this.sentences.length; i++) {
        if (this.sentences[i].text.toLowerCase().includes(lower)) {
          this.results.push(i);
        }
      }
    }
    this.updateCount();
    this.renderList();
    if (this.results.length) this.goTo(0);
  }

  next() {
    if (!this.results.length) return;
    this.goTo((this.resultIdx + 1) % this.results.length);
  }

  prev() {
    if (!this.results.length) return;
    this.goTo((this.resultIdx - 1 + this.results.length) % this.results.length);
  }

  /** Draw search-match highlights on a canvas. Call after waveform render. */
  drawOverlay(ctx: CanvasRenderingContext2D, w: number, h: number) {
    if (!this.results.length) return;
    const cur = this.resultIdx >= 0 ? this.results[this.resultIdx] : -1;
    for (const idx of this.results) {
      const s = this.sentences[idx];
      const x1 = this.tl.timeToX(s.start / 1000, w);
      const x2 = this.tl.timeToX(s.end / 1000, w);
      const left = Math.max(0, x1), right = Math.min(w, x2);
      if (right <= left) continue;
      ctx.fillStyle = idx === cur ? 'rgba(255,152,0,0.3)' : 'rgba(255,235,59,0.15)';
      ctx.fillRect(left, 0, right - left, h);
    }
  }

  private goTo(idx: number) {
    this.resultIdx = idx;
    const s = this.sentences[this.results[idx]];
    const startSec = s.start / 1000;
    const endSec = s.end / 1000;
    const span = this.tl.viewEnd - this.tl.viewStart;
    const center = (startSec + endSec) / 2;
    this.tl.setView(center - span / 2, center + span / 2);
    this.tl.setCursor(startSec);
    this.updateCount();
    this.renderList();
    const el = this.listEl.querySelector(`[data-idx="${this.results[idx]}"]`);
    el?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  private updateCount() {
    if (!this.query) { this.countEl.textContent = ''; return; }
    this.countEl.textContent = this.results.length
      ? `${this.resultIdx + 1}/${this.results.length}`
      : 'No results';
  }

  renderList() {
    const matchSet = new Set(this.results);
    const curSent = this.resultIdx >= 0 ? this.results[this.resultIdx] : -1;
    const re = this.query ? new RegExp(escRe(this.query), 'gi') : null;
    let html = '';

    for (let i = 0; i < this.sentences.length; i++) {
      const s = this.sentences[i];
      const cls = i === curSent ? 'sentence current'
        : matchSet.has(i) ? 'sentence match' : 'sentence';
      const time = fmtTime(s.start / 1000);
      let txt = escHtml(s.text);
      if (re && matchSet.has(i)) txt = txt.replace(re, '<mark>$&</mark>');
      html += `<div class="${cls}" data-idx="${i}" data-start="${s.start}" data-end="${s.end}">` +
        `<span class="time">${time}</span><span class="txt">${txt}</span></div>`;
    }
    this.listEl.innerHTML = html;
  }
}

function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}

function escRe(s: string) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }
function escHtml(s: string) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

import type { Timeline } from './timeline';

export interface CharTiming {
  char: string;
  start: number; // ms, -1 for punctuation without timestamp
  end: number;
}

export interface Sentence {
  text: string;
  start: number; // ms
  end: number; // ms
  chars: CharTiming[];
}

interface RawSentence {
  text: string; start: number; end: number;
  raw_text?: string; timestamp?: number[][];
}

export class ASRPanel {
  sentences: Sentence[] = [];
  results: number[] = [];
  resultIdx = -1;

  private query = '';
  private listEl: HTMLElement;
  private countEl: HTMLElement;
  private activeCharEl: HTMLElement | null = null;
  private activeCharRange: [number, number] = [-1, -1];

  constructor(
    container: HTMLElement,
    private tl: Timeline,
  ) {
    this.listEl = container.querySelector('#transcript-list')!;
    this.countEl = container.querySelector('#search-count')!;
  }

  load(json: unknown[]) {
    this.sentences = [];
    for (const rec of json as Array<{ sentences?: RawSentence[] }>) {
      if (rec.sentences) {
        for (const s of rec.sentences) {
          const chars = buildChars(s.text ?? '', s.raw_text ?? '', s.timestamp ?? []);
          this.sentences.push({ text: s.text ?? '', start: s.start, end: s.end, chars });
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

  /** Highlight the character at the given cursor time (karaoke). */
  highlightAt(timeMs: number) {
    if (this.activeCharEl && timeMs >= this.activeCharRange[0] && timeMs < this.activeCharRange[1]) {
      return; // still within current character
    }
    this.clearHighlight();

    const si = this.findSentenceAt(timeMs);
    if (si < 0) return;

    const s = this.sentences[si];
    let charIdx = -1;
    for (let ci = 0; ci < s.chars.length; ci++) {
      const c = s.chars[ci];
      if (c.start >= 0 && timeMs >= c.start && timeMs < c.end) { charIdx = ci; break; }
    }
    if (charIdx < 0) return;

    const sentEl = this.listEl.querySelector(`[data-idx="${si}"]`);
    if (!sentEl) return;
    const txtEl = sentEl.querySelector('.txt');
    if (!txtEl) return;
    const el = txtEl.children[charIdx] as HTMLElement | undefined;
    if (!el) return;

    el.classList.add('char-active');
    this.activeCharEl = el;
    this.activeCharRange = [s.chars[charIdx].start, s.chars[charIdx].end];
    sentEl.scrollIntoView?.({ block: 'nearest', behavior: 'auto' });
  }

  clearHighlight() {
    if (this.activeCharEl) {
      this.activeCharEl.classList.remove('char-active');
      this.activeCharEl = null;
      this.activeCharRange = [-1, -1];
    }
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

  /** Draw ASR segment boundaries and text labels on the canvas. */
  drawLabels(ctx: CanvasRenderingContext2D, w: number, h: number) {
    if (!this.sentences.length) return;
    const vStart = this.tl.viewStart;
    const vEnd = this.tl.viewEnd;
    const vSpan = vEnd - vStart;

    const fontSize = 11;
    ctx.font = `${fontSize}px monospace`;
    ctx.textBaseline = 'top';
    const rowH = fontSize + 6;
    const labelY = h - rowH;

    for (let si = 0; si < this.sentences.length; si++) {
      const s = this.sentences[si];
      const startSec = s.start / 1000;
      const endSec = s.end / 1000;
      if (endSec < vStart || startSec > vEnd) continue;

      const rawX1 = ((startSec - vStart) / vSpan) * w;
      const rawX2 = ((endSec - vStart) / vSpan) * w;
      const x1 = Math.max(0, rawX1);
      const x2 = Math.min(w, rawX2);
      const regionW = x2 - x1;

      // --- Alternating sentence background (full height) ---
      const sentColor = si % 2 === 0
        ? 'rgba(79, 195, 247, 0.06)'
        : 'rgba(255, 183, 77, 0.06)';
      ctx.fillStyle = sentColor;
      ctx.fillRect(x1, 0, regionW, labelY);

      // --- Per-character shading + gap highlighting when zoomed in ---
      const perChar = s.chars.length > 0 && regionW / s.text.length > 10;
      if (perChar) {
        let ci = 0;
        let prevEndMs = s.start; // track end of previous char to find gaps
        for (const c of s.chars) {
          if (c.start < 0) continue;
          const cEndMs = c.end >= 0 ? c.end : c.start;

          // Gap before this character
          if (c.start > prevEndMs) {
            const gx1 = Math.max(0, ((prevEndMs / 1000 - vStart) / vSpan) * w);
            const gx2 = Math.min(w, ((c.start / 1000 - vStart) / vSpan) * w);
            if (gx2 > gx1) {
              ctx.fillStyle = 'rgba(244, 67, 54, 0.12)';
              ctx.fillRect(gx1, 0, gx2 - gx1, labelY);
            }
          }

          // Character region
          const cx1 = ((c.start / 1000 - vStart) / vSpan) * w;
          const cx2 = ((cEndMs / 1000 - vStart) / vSpan) * w;
          const cl = Math.max(0, cx1), cr = Math.min(w, cx2);
          if (cr > cl) {
            ctx.fillStyle = ci % 2 === 0
              ? 'rgba(79, 195, 247, 0.08)'
              : 'rgba(255, 183, 77, 0.08)';
            ctx.fillRect(cl, 0, cr - cl, labelY);
          }
          prevEndMs = cEndMs;
          ci++;
        }
        // Gap after last character to sentence end
        if (prevEndMs < s.end) {
          const gx1 = Math.max(0, ((prevEndMs / 1000 - vStart) / vSpan) * w);
          const gx2 = Math.min(w, ((endSec - vStart) / vSpan) * w);
          if (gx2 > gx1) {
            ctx.fillStyle = 'rgba(244, 67, 54, 0.12)';
            ctx.fillRect(gx1, 0, gx2 - gx1, labelY);
          }
        }
      }

      // --- Sentence boundary lines (full height) ---
      ctx.setLineDash([3, 3]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = 'rgba(255, 235, 59, 0.4)';
      if (rawX1 >= 0 && rawX1 <= w) {
        ctx.beginPath();
        ctx.moveTo(rawX1, 0);
        ctx.lineTo(rawX1, h);
        ctx.stroke();
      }
      if (rawX2 >= 0 && rawX2 <= w) {
        ctx.beginPath();
        ctx.moveTo(rawX2, 0);
        ctx.lineTo(rawX2, h);
        ctx.stroke();
      }
      ctx.setLineDash([]);

      // --- Per-character boundary lines when zoomed in ---
      if (perChar) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
        ctx.lineWidth = 0.5;
        for (const c of s.chars) {
          if (c.start < 0) continue;
          const cx = ((c.start / 1000 - vStart) / vSpan) * w;
          if (cx > x1 + 1 && cx <= x2) {
            ctx.beginPath();
            ctx.moveTo(cx, 0);
            ctx.lineTo(cx, h);
            ctx.stroke();
          }
        }
      }

      if (regionW < 4) continue;

      // --- Text label bar at bottom ---
      if (perChar) {
        // Alternating background per character + gap highlights
        let ci = 0;
        let prevEnd = s.start;
        for (const c of s.chars) {
          if (c.start < 0) continue;
          const cEnd = c.end >= 0 ? c.end : c.start;
          // Gap in label bar
          if (c.start > prevEnd) {
            const gx1 = Math.max(x1, ((prevEnd / 1000 - vStart) / vSpan) * w);
            const gx2 = Math.min(x2, ((c.start / 1000 - vStart) / vSpan) * w);
            if (gx2 > gx1) {
              ctx.fillStyle = 'rgba(244, 67, 54, 0.35)';
              ctx.fillRect(gx1, labelY, gx2 - gx1, rowH);
            }
          }
          // Character cell
          const cx1c = Math.max(x1, ((c.start / 1000 - vStart) / vSpan) * w);
          const cx2c = Math.min(x2, ((cEnd / 1000 - vStart) / vSpan) * w);
          if (cx2c > cx1c) {
            ctx.fillStyle = ci % 2 === 0
              ? 'rgba(30, 60, 80, 0.85)'
              : 'rgba(50, 40, 30, 0.85)';
            ctx.fillRect(cx1c, labelY, cx2c - cx1c, rowH);
          }
          prevEnd = cEnd;
          ci++;
        }
        // Trailing gap
        if (prevEnd < s.end) {
          const gx1 = Math.max(x1, ((prevEnd / 1000 - vStart) / vSpan) * w);
          const gx2 = Math.min(x2, ((endSec - vStart) / vSpan) * w);
          if (gx2 > gx1) {
            ctx.fillStyle = 'rgba(244, 67, 54, 0.35)';
            ctx.fillRect(gx1, labelY, gx2 - gx1, rowH);
          }
        }
      } else {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';
        ctx.fillRect(x1, labelY, regionW, rowH);
      }

      // Clip text to region
      ctx.save();
      ctx.beginPath();
      ctx.rect(x1 + 1, labelY, regionW - 2, rowH);
      ctx.clip();

      ctx.fillStyle = '#ddd';
      if (perChar) {
        let lastEnd = s.start;
        for (const c of s.chars) {
          const cStart = c.start >= 0 ? c.start : lastEnd;
          const cx = ((cStart / 1000 - vStart) / vSpan) * w;
          ctx.fillText(c.char, cx + 1, labelY + 3);
          if (c.end >= 0) lastEnd = c.end;
        }
      } else {
        ctx.fillText(s.text, x1 + 2, labelY + 3);
      }

      ctx.restore();
    }
  }

  /** Zoom viewport to fit a sentence with some padding. */
  zoomToSentence(sentIdx: number) {
    const s = this.sentences[sentIdx];
    if (!s) return;
    const startSec = s.start / 1000;
    const endSec = s.end / 1000;
    const dur = endSec - startSec;
    const pad = dur * 0.15; // 15% padding on each side
    this.tl.setView(startSec - pad, endSec + pad);
    this.tl.setCursor(startSec);
  }

  private goTo(idx: number) {
    this.resultIdx = idx;
    const sentIdx = this.results[idx];
    this.zoomToSentence(sentIdx);
    this.updateCount();
    this.renderList();
    const el = this.listEl.querySelector(`[data-idx="${sentIdx}"]`);
    el?.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  private updateCount() {
    if (!this.query) { this.countEl.textContent = ''; return; }
    this.countEl.textContent = this.results.length
      ? `${this.resultIdx + 1}/${this.results.length}`
      : 'No results';
  }

  renderList() {
    this.clearHighlight();
    const matchSet = new Set(this.results);
    const curSent = this.resultIdx >= 0 ? this.results[this.resultIdx] : -1;
    let html = '';

    for (let i = 0; i < this.sentences.length; i++) {
      const s = this.sentences[i];
      const cls = i === curSent ? 'sentence current'
        : matchSet.has(i) ? 'sentence match' : 'sentence';
      const time = fmtTime(s.start / 1000);

      const hlSet = this.query && matchSet.has(i)
        ? matchHlSet(s.text, this.query) : null;

      let txtHtml = '';
      for (let ci = 0; ci < s.chars.length; ci++) {
        const c = s.chars[ci];
        const hl = hlSet?.has(ci) ? ' search-hl' : '';
        const ch = escHtml(c.char);
        if (c.start >= 0) {
          txtHtml += `<span class="char${hl}" data-cs="${c.start}" data-ce="${c.end}" title="${fmtMs(c.start)} \u2192 ${fmtMs(c.end)}">${ch}</span>`;
        } else {
          txtHtml += `<span class="punc${hl}">${ch}</span>`;
        }
      }

      html += `<div class="${cls}" data-idx="${i}" data-start="${s.start}" data-end="${s.end}">` +
        `<span class="time">${time}</span><span class="txt">${txtHtml}</span></div>`;
    }
    this.listEl.innerHTML = html;
  }

  private findSentenceAt(timeMs: number): number {
    for (let i = 0; i < this.sentences.length; i++) {
      if (timeMs >= this.sentences[i].start && timeMs <= this.sentences[i].end) return i;
    }
    return -1;
  }
}

function buildChars(text: string, rawText: string, timestamps: number[][]): CharTiming[] {
  const chars: CharTiming[] = [];
  let ri = 0;
  for (let i = 0; i < text.length; i++) {
    if (ri < rawText.length && text[i] === rawText[ri]) {
      const ts = timestamps[ri];
      chars.push({ char: text[i], start: ts?.[0] ?? -1, end: ts?.[1] ?? -1 });
      ri++;
    } else {
      chars.push({ char: text[i], start: -1, end: -1 });
    }
  }
  return chars;
}

function matchHlSet(text: string, query: string): Set<number> {
  const set = new Set<number>();
  const lower = text.toLowerCase();
  const q = query.toLowerCase();
  let pos = 0;
  for (;;) {
    const idx = lower.indexOf(q, pos);
    if (idx === -1) break;
    for (let i = idx; i < idx + q.length; i++) set.add(i);
    pos = idx + 1;
  }
  return set;
}

function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}

function fmtMs(ms: number): string {
  const sec = ms / 1000;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`;
}

function escHtml(s: string) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }

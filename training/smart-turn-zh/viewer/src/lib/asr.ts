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

// ---- Data parsing ----

export function parseSentences(json: unknown[]): Sentence[] {
  const sentences: Sentence[] = [];
  for (const rec of json as Array<{ sentences?: RawSentence[] }>) {
    if (rec.sentences) {
      for (const s of rec.sentences) {
        const chars = buildChars(s.text ?? '', s.raw_text ?? '', s.timestamp ?? []);
        sentences.push({ text: s.text ?? '', start: s.start, end: s.end, chars });
      }
    }
  }
  return sentences;
}

export function searchSentences(sentences: Sentence[], query: string): number[] {
  if (!query) return [];
  const lower = query.toLowerCase();
  const results: number[] = [];
  for (let i = 0; i < sentences.length; i++) {
    if (sentences[i].text.toLowerCase().includes(lower)) results.push(i);
  }
  return results;
}

export function zoomToSentence(tl: Timeline, sentences: Sentence[], sentIdx: number): void {
  const s = sentences[sentIdx];
  if (!s) return;
  const startSec = s.start / 1000;
  const endSec = s.end / 1000;
  const dur = endSec - startSec;
  const pad = dur * 0.15;
  tl.setView(startSec - pad, endSec + pad);
  tl.setCursor(startSec);
}

// ---- Canvas drawing ----

export function drawASROverlay(
  ctx: CanvasRenderingContext2D, w: number, h: number,
  tl: Timeline, sentences: Sentence[],
  results: number[], currentResultIdx: number,
): void {
  if (!results.length) return;
  const cur = currentResultIdx >= 0 ? results[currentResultIdx] : -1;
  for (const idx of results) {
    const s = sentences[idx];
    const x1 = tl.timeToX(s.start / 1000, w);
    const x2 = tl.timeToX(s.end / 1000, w);
    const left = Math.max(0, x1), right = Math.min(w, x2);
    if (right <= left) continue;
    ctx.fillStyle = idx === cur ? 'rgba(255,152,0,0.3)' : 'rgba(255,235,59,0.15)';
    ctx.fillRect(left, 0, right - left, h);
  }
}

export function drawASRLabels(
  ctx: CanvasRenderingContext2D, w: number, h: number,
  tl: Timeline, sentences: Sentence[],
): void {
  if (!sentences.length) return;
  const vStart = tl.viewStart;
  const vEnd = tl.viewEnd;
  const vSpan = vEnd - vStart;

  const fontSize = 11;
  ctx.font = `${fontSize}px monospace`;
  ctx.textBaseline = 'top';
  const rowH = fontSize + 6;
  const labelY = h - rowH;

  for (let si = 0; si < sentences.length; si++) {
    const s = sentences[si];
    const startSec = s.start / 1000;
    const endSec = s.end / 1000;
    if (endSec < vStart || startSec > vEnd) continue;

    const rawX1 = ((startSec - vStart) / vSpan) * w;
    const rawX2 = ((endSec - vStart) / vSpan) * w;
    const x1 = Math.max(0, rawX1);
    const x2 = Math.min(w, rawX2);
    const regionW = x2 - x1;

    // Alternating sentence background
    const sentColor = si % 2 === 0
      ? 'rgba(79, 195, 247, 0.06)'
      : 'rgba(255, 183, 77, 0.06)';
    ctx.fillStyle = sentColor;
    ctx.fillRect(x1, 0, regionW, labelY);

    // Per-character shading + gap highlighting when zoomed in
    const perChar = s.chars.length > 0 && regionW / s.text.length > 10;
    if (perChar) {
      let ci = 0;
      let prevEndMs = s.start;
      for (const c of s.chars) {
        if (c.start < 0) continue;
        const cEndMs = c.end >= 0 ? c.end : c.start;

        if (c.start > prevEndMs) {
          const gx1 = Math.max(0, ((prevEndMs / 1000 - vStart) / vSpan) * w);
          const gx2 = Math.min(w, ((c.start / 1000 - vStart) / vSpan) * w);
          if (gx2 > gx1) {
            ctx.fillStyle = 'rgba(244, 67, 54, 0.12)';
            ctx.fillRect(gx1, 0, gx2 - gx1, labelY);
          }
        }

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
      if (prevEndMs < s.end) {
        const gx1 = Math.max(0, ((prevEndMs / 1000 - vStart) / vSpan) * w);
        const gx2 = Math.min(w, ((endSec - vStart) / vSpan) * w);
        if (gx2 > gx1) {
          ctx.fillStyle = 'rgba(244, 67, 54, 0.12)';
          ctx.fillRect(gx1, 0, gx2 - gx1, labelY);
        }
      }
    }

    // Sentence boundary lines
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

    // Per-character boundary lines
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

    // Text label bar at bottom
    if (perChar) {
      let ci = 0;
      let prevEnd = s.start;
      for (const c of s.chars) {
        if (c.start < 0) continue;
        const cEnd = c.end >= 0 ? c.end : c.start;
        if (c.start > prevEnd) {
          const gx1 = Math.max(x1, ((prevEnd / 1000 - vStart) / vSpan) * w);
          const gx2 = Math.min(x2, ((c.start / 1000 - vStart) / vSpan) * w);
          if (gx2 > gx1) {
            ctx.fillStyle = 'rgba(244, 67, 54, 0.35)';
            ctx.fillRect(gx1, labelY, gx2 - gx1, rowH);
          }
        }
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

// ---- Utilities ----

export function matchHlSet(text: string, query: string): Set<number> {
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

export function fmtTime(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}

export function fmtMs(ms: number): string {
  const sec = ms / 1000;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}`;
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

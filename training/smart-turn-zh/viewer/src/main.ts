import { Timeline } from './timeline';
import { AudioStore } from './audio';
import { WaveformRenderer } from './waveform';
import { VADRenderer } from './vad';
import { ASRPanel } from './asr';
import './style.css';

// --- State ---

const tl = new Timeline();
const audio = new AudioStore();

// --- DOM ---

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

const wavInput = $<HTMLInputElement>('wav-input');
const vadInput = $<HTMLInputElement>('vad-input');
const asrInput = $<HTMLInputElement>('asr-input');
const wavStatus = $('wav-status');
const vadStatus = $('vad-status');
const asrStatus = $('asr-status');
const chSel = $<HTMLSelectElement>('channel-select');
const playBtn = $<HTMLButtonElement>('play-btn');
const timeDisp = $('time-display');
const wfCanvas = $<HTMLCanvasElement>('waveform-canvas');
const vadCanvas = $<HTMLCanvasElement>('vad-canvas');
const mmCanvas = $<HTMLCanvasElement>('minimap-canvas');
const searchIn = $<HTMLInputElement>('search-input');
const prevBtn = $<HTMLButtonElement>('prev-btn');
const nextBtn = $<HTMLButtonElement>('next-btn');
const vadEntry = $<HTMLInputElement>('vad-entry');
const vadExit = $<HTMLInputElement>('vad-exit');
const dropEl = $('drop-overlay');

// --- Renderers ---

const wf = new WaveformRenderer(wfCanvas, tl, audio);
const vad = new VADRenderer(vadCanvas, tl);
const asr = new ASRPanel($('asr-panel'), tl);

// --- Playback ---

let actx: AudioContext | null = null;
let srcNode: AudioBufferSourceNode | null = null;
let playing = false;
let playT0 = 0;
let playOff = 0;
let playBuf: AudioBuffer | null = null;
let playChannel = -1;

function ensurePlayBuf(ch: number): AudioBuffer | null {
  if (playBuf && playChannel === ch) return playBuf;
  playBuf = audio.createAudioBuffer(ch);
  playChannel = ch;
  return playBuf;
}

function play(offset: number) {
  if (!audio.raw) return;
  const buf = ensurePlayBuf(wf.channel);
  if (!buf) return;
  actx = new AudioContext({ sampleRate: audio.sampleRate });
  srcNode = actx.createBufferSource();
  srcNode.buffer = buf;
  srcNode.connect(actx.destination);
  srcNode.start(0, offset);
  srcNode.onended = () => stop();
  playT0 = actx.currentTime;
  playOff = offset;
  playing = true;
  playBtn.innerHTML = '&#9646;&#9646;';
  rafLoop();
}

function stop() {
  if (srcNode) {
    srcNode.onended = null;
    try { srcNode.stop(); } catch { /* already stopped */ }
    srcNode = null;
  }
  if (actx) {
    tl.cursor = curTime();
    actx.close();
    actx = null;
  }
  playing = false;
  playBtn.innerHTML = '&#9654;';
  renderAll();
}

function curTime(): number {
  if (!actx) return tl.cursor;
  return playOff + (actx.currentTime - playT0);
}

function rafLoop() {
  if (!playing) return;
  const t = curTime();
  if (t >= tl.duration) { stop(); return; }
  tl.cursor = t;
  // Auto-scroll when cursor exits viewport
  if (t > tl.viewEnd) {
    const span = tl.viewEnd - tl.viewStart;
    tl.viewStart = t;
    tl.viewEnd = t + span;
  }
  renderAll();
  requestAnimationFrame(rafLoop);
}

// --- File loading ---

wavInput.onchange = () => wavInput.files?.[0] && loadWav(wavInput.files[0]);
vadInput.onchange = () => vadInput.files?.[0] && loadVad(vadInput.files[0]);
asrInput.onchange = () => asrInput.files?.[0] && loadAsr(asrInput.files[0]);

async function loadWav(f: File) {
  wavStatus.textContent = '...';
  try {
    await audio.load(f);
    tl.sampleRate = audio.sampleRate;
    tl.setDuration(audio.duration);
    chSel.innerHTML = '<option value="-1">All</option>';
    for (let i = 0; i < audio.channelCount; i++) {
      chSel.innerHTML += `<option value="${i}">Ch ${i + 1}</option>`;
    }
    chSel.disabled = false;
    playBtn.disabled = false;
    playBuf = null;
    wavStatus.textContent = '\u2713';
  } catch (e) {
    wavStatus.textContent = '\u2717';
    console.error(e);
  }
}

async function loadVad(f: File) {
  vadStatus.textContent = '...';
  try {
    vad.load(await f.arrayBuffer());
    vadStatus.textContent = '\u2713';
    renderAll();
  } catch (e) {
    vadStatus.textContent = '\u2717';
    console.error(e);
  }
}

async function loadAsr(f: File) {
  asrStatus.textContent = '...';
  try {
    const json = JSON.parse(await f.text());
    asr.load(Array.isArray(json) ? json : [json]);
    asrStatus.textContent = '\u2713';
    prevBtn.disabled = false;
    nextBtn.disabled = false;
  } catch (e) {
    asrStatus.textContent = '\u2717';
    console.error(e);
  }
}

// --- Drag & drop ---

document.ondragover = (e) => { e.preventDefault(); dropEl.hidden = false; };
dropEl.ondragleave = () => { dropEl.hidden = true; };
document.ondrop = (e) => {
  e.preventDefault();
  dropEl.hidden = true;
  for (const f of e.dataTransfer?.files ?? []) {
    if (f.name.endsWith('.wav')) loadWav(f);
    else if (f.name.endsWith('.npy')) loadVad(f);
    else if (f.name.endsWith('.json')) loadAsr(f);
  }
};

// --- Controls ---

chSel.onchange = () => {
  wf.channel = +chSel.value;
  playBuf = null;
  renderAll();
};

playBtn.onclick = () => playing ? stop() : play(tl.cursor);

vadEntry.oninput = () => { vad.entryThreshold = +vadEntry.value; renderAll(); };
vadExit.oninput = () => { vad.exitThreshold = +vadExit.value; renderAll(); };

searchIn.oninput = () => { asr.search(searchIn.value); renderAll(); };
prevBtn.onclick = () => { asr.prev(); renderAll(); };
nextBtn.onclick = () => { asr.next(); renderAll(); };

// ASR sentence click to seek
$('transcript-list').onclick = (e) => {
  const el = (e.target as HTMLElement).closest('.sentence') as HTMLElement | null;
  if (!el) return;
  const t = +el.dataset.start! / 1000;
  tl.setCursor(t);
  if (playing) { stop(); play(t); }
};

// --- Canvas interaction (pan / zoom / seek) ---

for (const cvs of [wfCanvas, vadCanvas]) {
  cvs.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = cvs.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    tl.zoom(e.deltaY > 0 ? 1.25 : 0.8, frac);
  }, { passive: false });

  let drag = false, startX = 0, startVS = 0, moved = false;

  cvs.onmousedown = (e) => {
    drag = true; moved = false; startX = e.clientX; startVS = tl.viewStart;
  };
  window.addEventListener('mousemove', (e) => {
    if (!drag) return;
    const dx = e.clientX - startX;
    if (Math.abs(dx) > 3) moved = true;
    if (!moved) return;
    const rect = cvs.getBoundingClientRect();
    const span = tl.viewEnd - tl.viewStart;
    const dt = -(dx / rect.width) * span;
    tl.setView(startVS + dt, startVS + dt + span);
  });
  window.addEventListener('mouseup', (e) => {
    if (!drag) return;
    drag = false;
    if (!moved) {
      const rect = cvs.getBoundingClientRect();
      const t = tl.xToTime(e.clientX - rect.left, rect.width);
      tl.setCursor(t);
      if (playing) { stop(); play(t); }
    }
  });
}

// Minimap click to navigate
mmCanvas.onclick = (e) => {
  const rect = mmCanvas.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  const t = frac * tl.duration;
  const span = tl.viewEnd - tl.viewStart;
  tl.setView(t - span / 2, t + span / 2);
};

// --- Keyboard shortcuts ---

document.onkeydown = (e) => {
  // Let search input handle its own keys (except Escape)
  if (e.target === searchIn && e.key !== 'Escape') {
    if (e.key === 'Enter') {
      e.preventDefault();
      e.shiftKey ? asr.prev() : asr.next();
      renderAll();
    }
    return;
  }
  switch (e.key) {
    case ' ':
      e.preventDefault();
      playing ? stop() : play(tl.cursor);
      break;
    case 'f':
      e.preventDefault();
      searchIn.focus();
      break;
    case 'Escape':
      searchIn.blur();
      break;
    case '=': case '+':
      tl.zoom(0.67, 0.5);
      break;
    case '-':
      tl.zoom(1.5, 0.5);
      break;
    case 'ArrowLeft':
      tl.pan(-(tl.viewEnd - tl.viewStart) * 0.1);
      break;
    case 'ArrowRight':
      tl.pan((tl.viewEnd - tl.viewStart) * 0.1);
      break;
    case '0':
      wf.channel = -1; chSel.value = '-1'; playBuf = null; renderAll();
      break;
    default:
      if (e.key >= '1' && e.key <= '9') {
        const ch = +e.key - 1;
        if (ch < audio.channelCount) {
          wf.channel = ch; chSel.value = String(ch); playBuf = null; renderAll();
        }
      }
  }
};

// --- Render ---

function renderAll() {
  wf.render();
  vad.render();
  renderMinimap();

  // ASR search overlay on waveform canvas
  if (asr.results.length) {
    const dpr = devicePixelRatio;
    const w = wfCanvas.width / dpr;
    const h = wfCanvas.height / dpr;
    const ctx = wfCanvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    asr.drawOverlay(ctx, w, h);
  }

  timeDisp.textContent = `${fmt(tl.cursor)} / ${fmt(tl.duration)}`;
}

function renderMinimap() {
  if (!audio.raw) return;
  const dpr = devicePixelRatio;
  const rect = mmCanvas.getBoundingClientRect();
  mmCanvas.width = rect.width * dpr;
  mmCanvas.height = rect.height * dpr;
  const ctx = mmCanvas.getContext('2d')!;
  const w = rect.width, h = rect.height;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  // Coarsest LOD waveform overview
  const levels = audio.getLOD(wf.channel);
  if (levels.length) {
    const lv = levels[levels.length - 1];
    const mid = h / 2, amp = mid;
    ctx.fillStyle = '#1e3a4a';
    ctx.beginPath();
    for (let x = 0; x < w; x++) {
      const b = Math.min(Math.floor((x / w) * lv.max.length), lv.max.length - 1);
      x === 0 ? ctx.moveTo(x, mid - lv.max[b] * amp) : ctx.lineTo(x, mid - lv.max[b] * amp);
    }
    for (let x = w - 1; x >= 0; x--) {
      const b = Math.min(Math.floor((x / w) * lv.min.length), lv.min.length - 1);
      ctx.lineTo(x, mid - lv.min[b] * amp);
    }
    ctx.closePath();
    ctx.fill();
  }

  // Viewport indicator
  const x1 = (tl.viewStart / tl.duration) * w;
  const x2 = (tl.viewEnd / tl.duration) * w;
  ctx.fillStyle = 'rgba(79,195,247,0.15)';
  ctx.fillRect(x1, 0, x2 - x1, h);
  ctx.strokeStyle = '#4fc3f7';
  ctx.lineWidth = 1;
  ctx.strokeRect(x1 + 0.5, 0.5, x2 - x1 - 1, h - 1);
}

function fmt(sec: number): string {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, '0')}:${s.toFixed(1).padStart(4, '0')}`;
}

// --- Timeline update listener (batched via rAF) ---

let renderPending = false;
tl.onUpdate(() => {
  if (playing) return; // playback loop handles rendering
  if (renderPending) return;
  renderPending = true;
  requestAnimationFrame(() => { renderPending = false; renderAll(); });
});

// --- Resize ---

const ro = new ResizeObserver(() => { wf.resize(); vad.resize(); renderAll(); });
ro.observe(wfCanvas.parentElement!);
ro.observe(vadCanvas.parentElement!);
wf.resize();
vad.resize();

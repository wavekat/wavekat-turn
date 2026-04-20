import { useRef, useState, useCallback, useEffect } from 'react';
import { Timeline } from './lib/timeline';
import { AudioStore } from './lib/audio';
import type { WaveformScale } from './lib/waveform';
import {
  type Sentence,
  parseSentences, searchSentences, zoomToSentence,
} from './lib/asr';
import { Toolbar } from './components/Toolbar';
import { WaveformTrack } from './components/WaveformTrack';
import { VADTrack } from './components/VADTrack';
import { SpectrogramTrack } from './components/SpectrogramTrack';
import { ResizeHandle } from './components/ResizeHandle';
import { Minimap } from './components/Minimap';
import { ASRPanel } from './components/ASRPanel';
import { usePlayback } from './hooks/usePlayback';

export function App() {
  // Stable singleton instances
  const [tl] = useState(() => new Timeline());
  const [audio] = useState(() => new AudioStore());

  // File loading status
  const [wavStatus, setWavStatus] = useState('');
  const [vadStatus, setVadStatus] = useState('');
  const [asrStatus, setAsrStatus] = useState('');

  // Audio state
  const [channelCount, setChannelCount] = useState(0);
  const [channel, setChannel] = useState(-1);
  const [scale, setScale] = useState<WaveformScale>('dB');
  const [gainValue, setGainValue] = useState(10.0);

  // VAD state
  const [vadBuffer, setVadBuffer] = useState<ArrayBuffer | null>(null);
  const [vadEntry, setVadEntry] = useState(0.3);
  const [vadExit, setVadExit] = useState(0.1);

  // ASR state
  const [sentences, setSentences] = useState<Sentence[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<number[]>([]);
  const [searchIdx, setSearchIdx] = useState(-1);

  // Drop overlay
  const [showDrop, setShowDrop] = useState(false);

  // ---- Layout persistence & resize handlers ----

  const LAYOUT_KEY = 'viewer-layout';

  const [resizeTrack] = useState(() => {
    const ids = ['waveform-track', 'vad-track', 'spectrogram-track'];
    const MIN = 30;

    const save = () => {
      const els = ids.map((id) => document.getElementById(id));
      const panel = document.getElementById('asr-panel');
      if (els.some((e) => !e) || !panel) return;
      const layout = {
        trackFlex: els.map((e) => e!.offsetHeight),
        asrWidth: panel.offsetWidth,
      };
      localStorage.setItem(LAYOUT_KEY, JSON.stringify(layout));
    };

    const resize = (aboveIdx: number, belowIdx: number) => (delta: number) => {
      const els = ids.map((id) => document.getElementById(id));
      if (els.some((e) => !e)) return;
      const heights = els.map((e) => e!.offsetHeight);

      heights[aboveIdx] += delta;
      heights[belowIdx] -= delta;
      if (heights[aboveIdx] < MIN || heights[belowIdx] < MIN) return;

      els.forEach((e, i) => { e!.style.flex = `${heights[i]} 0 0px`; });
      save();
    };

    return {
      wfVad: resize(0, 1),
      vadSpec: resize(1, 2),
      asr: (delta: number) => {
        const panel = document.getElementById('asr-panel');
        if (!panel) return;
        panel.style.width = `${Math.max(200, Math.min(600, panel.offsetWidth - delta))}px`;
        save();
      },
    };
  });

  // Restore saved layout on mount
  useEffect(() => {
    try {
      const raw = localStorage.getItem(LAYOUT_KEY);
      if (!raw) return;
      const layout = JSON.parse(raw) as { trackFlex?: number[]; asrWidth?: number };
      const ids = ['waveform-track', 'vad-track', 'spectrogram-track'];
      if (layout.trackFlex?.length === 3) {
        ids.forEach((id, i) => {
          const el = document.getElementById(id);
          if (el) el.style.flex = `${layout.trackFlex![i]} 0 0px`;
        });
      }
      if (layout.asrWidth) {
        const panel = document.getElementById('asr-panel');
        if (panel) panel.style.width = `${layout.asrWidth}px`;
      }
    } catch { /* ignore corrupt data */ }
  }, []);

  // Playback
  const { playing, play, stop, setGain, invalidateBuffer } = usePlayback(tl, audio);

  // Refs for stable callbacks that need current values
  const playingRef = useRef(playing);
  const channelRef = useRef(channel);
  const playRef = useRef(play);
  const stopRef = useRef(stop);
  playingRef.current = playing;
  channelRef.current = channel;
  playRef.current = play;
  stopRef.current = stop;

  // Sentences ref for search callbacks
  const sentencesRef = useRef(sentences);
  sentencesRef.current = sentences;

  // Search state refs for next/prev
  const searchResultsRef = useRef(searchResults);
  const searchIdxRef = useRef(searchIdx);
  searchResultsRef.current = searchResults;
  searchIdxRef.current = searchIdx;

  // ---- Handlers ----

  const handleSeek = useCallback((t: number) => {
    tl.setCursor(t);
    if (playingRef.current) {
      stopRef.current();
      playRef.current(t, channelRef.current);
    }
  }, [tl]);

  const handlePlayToggle = useCallback(() => {
    if (playingRef.current) {
      stopRef.current();
    } else {
      playRef.current(tl.cursor, channelRef.current);
    }
  }, [tl]);

  const handleChannelChange = useCallback((ch: number) => {
    setChannel(ch);
    invalidateBuffer();
  }, [invalidateBuffer]);

  const handleGainChange = useCallback((v: number) => {
    setGainValue(v);
    setGain(v);
  }, [setGain]);

  // ---- File loading ----

  const loadWav = useCallback(async (f: File) => {
    setWavStatus('...');
    try {
      await audio.load(f);
      tl.sampleRate = audio.sampleRate;
      tl.setDuration(audio.duration);
      setChannelCount(audio.channelCount);
      setChannel(-1);
      invalidateBuffer();
      setWavStatus('\u2713');
    } catch (e) {
      setWavStatus('\u2717');
      console.error(e);
    }
  }, [audio, tl, invalidateBuffer]);

  const loadVad = useCallback(async (f: File) => {
    setVadStatus('...');
    try {
      setVadBuffer(await f.arrayBuffer());
      setVadStatus('\u2713');
    } catch (e) {
      setVadStatus('\u2717');
      console.error(e);
    }
  }, []);

  const loadAsr = useCallback(async (f: File) => {
    setAsrStatus('...');
    try {
      const json = JSON.parse(await f.text());
      const sents = parseSentences(Array.isArray(json) ? json : [json]);
      setSentences(sents);
      setSearchQuery('');
      setSearchResults([]);
      setSearchIdx(-1);
      setAsrStatus('\u2713');
    } catch (e) {
      setAsrStatus('\u2717');
      console.error(e);
    }
  }, []);

  const routeFile = useCallback((f: File) => {
    if (f.name.endsWith('.wav')) loadWav(f);
    else if (f.name.endsWith('.npy')) loadVad(f);
    else if (f.name.endsWith('.json')) loadAsr(f);
  }, [loadWav, loadVad, loadAsr]);

  const handleFilesSelected = useCallback((files: File[]) => {
    for (const f of files) routeFile(f);
  }, [routeFile]);

  // ---- Search ----

  const handleSearch = useCallback((q: string) => {
    setSearchQuery(q);
    const sents = sentencesRef.current;
    const results = searchSentences(sents, q);
    setSearchResults(results);
    if (results.length) {
      setSearchIdx(0);
      zoomToSentence(tl, sents, results[0]);
    } else {
      setSearchIdx(-1);
    }
  }, [tl]);

  const handleSearchNext = useCallback(() => {
    const results = searchResultsRef.current;
    if (!results.length) return;
    const next = (searchIdxRef.current + 1) % results.length;
    setSearchIdx(next);
    zoomToSentence(tl, sentencesRef.current, results[next]);
  }, [tl]);

  const handleSearchPrev = useCallback(() => {
    const results = searchResultsRef.current;
    if (!results.length) return;
    const prev = (searchIdxRef.current - 1 + results.length) % results.length;
    setSearchIdx(prev);
    zoomToSentence(tl, sentencesRef.current, results[prev]);
  }, [tl]);

  // ---- Drag & Drop ----

  useEffect(() => {
    const onDragOver = (e: DragEvent) => { e.preventDefault(); setShowDrop(true); };
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      setShowDrop(false);
      for (const f of e.dataTransfer?.files ?? []) routeFile(f);
    };
    document.addEventListener('dragover', onDragOver);
    document.addEventListener('drop', onDrop);
    return () => {
      document.removeEventListener('dragover', onDragOver);
      document.removeEventListener('drop', onDrop);
    };
  }, [routeFile]);

  // ---- Keyboard shortcuts ----

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const searchInput = document.getElementById('search-input');

      // Let search input handle its own keys (except Escape)
      if (e.target === searchInput && e.key !== 'Escape') return;

      switch (e.key) {
        case ' ':
          e.preventDefault();
          handlePlayToggle();
          break;
        case 'f':
          e.preventDefault();
          searchInput?.focus();
          break;
        case 'Escape':
          (document.activeElement as HTMLElement)?.blur();
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
          handleChannelChange(-1);
          break;
        default:
          if (e.key >= '1' && e.key <= '9') {
            const ch = +e.key - 1;
            if (ch < audio.channelCount) handleChannelChange(ch);
          }
      }
    };
    document.addEventListener('keydown', onKeyDown);
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [tl, audio, handlePlayToggle, handleChannelChange]);

  // ---- Render ----

  return (
    <div id="app">
      <Toolbar
        timeline={tl}
        onFilesSelected={handleFilesSelected}
        wavStatus={wavStatus}
        vadStatus={vadStatus}
        asrStatus={asrStatus}
        channelCount={channelCount}
        channel={channel}
        onChannelChange={handleChannelChange}
        playing={playing}
        canPlay={channelCount > 0}
        onPlayToggle={handlePlayToggle}
        gainValue={gainValue}
        onGainChange={handleGainChange}
      />

      <div id="main-area">
        <div id="tracks-column">
          <div id="tracks">
            <WaveformTrack
              timeline={tl}
              audio={audio}
              channel={channel}
              scale={scale}
              onScaleChange={setScale}
              sentences={sentences}
              searchResults={searchResults}
              searchResultIdx={searchIdx}
              onSeek={handleSeek}
            />
            <ResizeHandle direction="row" onDrag={resizeTrack.wfVad} />
            <VADTrack
              timeline={tl}
              vadBuffer={vadBuffer}
              entryThreshold={vadEntry}
              exitThreshold={vadExit}
              onEntryChange={setVadEntry}
              onExitChange={setVadExit}
              onSeek={handleSeek}
            />
            <ResizeHandle direction="row" onDrag={resizeTrack.vadSpec} />
            <SpectrogramTrack
              timeline={tl}
              audio={audio}
              channel={channel}
              onSeek={handleSeek}
            />
          </div>

          <Minimap timeline={tl} audio={audio} channel={channel} />
        </div>

        <ResizeHandle direction="col" onDrag={resizeTrack.asr} />

        <ASRPanel
          timeline={tl}
          sentences={sentences}
          searchQuery={searchQuery}
          searchResults={searchResults}
          searchResultIdx={searchIdx}
          onSearchChange={handleSearch}
          onNext={handleSearchNext}
          onPrev={handleSearchPrev}
          onSeek={handleSeek}
          playing={playing}
        />
      </div>

      <div id="drop-overlay" hidden={!showDrop} onDragLeave={() => setShowDrop(false)}>
        Drop WAV / NPY / JSON files
      </div>
    </div>
  );
}

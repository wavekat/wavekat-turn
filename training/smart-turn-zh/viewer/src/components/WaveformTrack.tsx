import { useRef, useEffect, useCallback, memo } from 'react';
import type { Timeline } from '../lib/timeline';
import type { AudioStore } from '../lib/audio';
import { WaveformRenderer, type WaveformScale } from '../lib/waveform';
import { drawASROverlay, drawASRLabels, type Sentence } from '../lib/asr';
import { useCanvasInteraction } from '../hooks/useCanvasInteraction';

interface WaveformTrackProps {
  timeline: Timeline;
  audio: AudioStore;
  channel: number;
  scale: WaveformScale;
  onScaleChange: (s: WaveformScale) => void;
  sentences: Sentence[];
  searchResults: number[];
  searchResultIdx: number;
  onSeek: (t: number) => void;
}

export const WaveformTrack = memo(function WaveformTrack({
  timeline, audio, channel, scale, onScaleChange,
  sentences, searchResults, searchResultIdx, onSeek,
}: WaveformTrackProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wfRef = useRef<WaveformRenderer | null>(null);

  // Store frequently-changing data in refs for the stable draw callback
  const channelRef = useRef(channel);
  const scaleRef = useRef(scale);
  const sentencesRef = useRef(sentences);
  const resultsRef = useRef(searchResults);
  const resultIdxRef = useRef(searchResultIdx);
  channelRef.current = channel;
  scaleRef.current = scale;
  sentencesRef.current = sentences;
  resultsRef.current = searchResults;
  resultIdxRef.current = searchResultIdx;

  // Stable draw function — reads everything from refs
  const draw = useCallback(() => {
    const wf = wfRef.current;
    const canvas = canvasRef.current;
    if (!wf || !canvas) return;

    wf.channel = channelRef.current;
    wf.scale = scaleRef.current;
    wf.render();

    // ASR overlays on top of waveform
    const sents = sentencesRef.current;
    if (sents.length) {
      const dpr = devicePixelRatio;
      const w = canvas.width / dpr;
      const h = canvas.height / dpr;
      const ctx = canvas.getContext('2d')!;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const results = resultsRef.current;
      if (results.length) {
        drawASROverlay(ctx, w, h, timeline, sents, results, resultIdxRef.current);
      }
      drawASRLabels(ctx, w, h, timeline, sents);
    }
  }, [timeline]);

  // Initialize renderer, subscribe to timeline, observe resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const wf = new WaveformRenderer(canvas, timeline, audio);
    wfRef.current = wf;
    wf.resize();
    draw();

    const unsub = timeline.onUpdate(draw);
    const ro = new ResizeObserver(() => { wf.resize(); draw(); });
    ro.observe(canvas.parentElement!);

    return () => {
      unsub();
      ro.disconnect();
      wfRef.current = null;
    };
  }, [timeline, audio, draw]);

  // Redraw when display props change
  useEffect(draw, [channel, scale, sentences, searchResults, searchResultIdx, draw]);

  useCanvasInteraction(canvasRef, timeline, onSeek);

  return (
    <div className="track" id="waveform-track">
      <span className="track-label">Waveform</span>
      <div id="wf-scale" className="scale-toggle">
        <button
          className={`scale-btn${scale === 'linear' ? ' active' : ''}`}
          onClick={() => onScaleChange('linear')}
        >
          Linear
        </button>
        <button
          className={`scale-btn${scale === 'dB' ? ' active' : ''}`}
          onClick={() => onScaleChange('dB')}
        >
          dB
        </button>
      </div>
      <canvas ref={canvasRef} />
    </div>
  );
});

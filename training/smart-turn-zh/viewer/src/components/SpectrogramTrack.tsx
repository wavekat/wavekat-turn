import { useRef, useEffect, useCallback, useState, memo } from 'react';
import type { Timeline } from '../lib/timeline';
import type { AudioStore } from '../lib/audio';
import { SpectrogramRenderer, type FreqScale } from '../lib/spectrogram';
import { useCanvasInteraction } from '../hooks/useCanvasInteraction';

interface SpectrogramTrackProps {
  timeline: Timeline;
  audio: AudioStore;
  channel: number;
  onSeek: (t: number) => void;
}

export const SpectrogramTrack = memo(function SpectrogramTrack({
  timeline, audio, channel, onSeek,
}: SpectrogramTrackProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<SpectrogramRenderer | null>(null);
  const [freqScale, setFreqScale] = useState<FreqScale>('mel');

  const channelRef = useRef(channel);
  const freqScaleRef = useRef(freqScale);
  channelRef.current = channel;
  freqScaleRef.current = freqScale;

  const draw = useCallback(() => {
    const r = rendererRef.current;
    if (!r) return;
    r.channel = channelRef.current;
    r.freqScale = freqScaleRef.current;
    r.render();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const r = new SpectrogramRenderer(canvas, timeline, audio);
    rendererRef.current = r;
    r.onTileReady = draw;
    r.resize();
    draw();

    const unsub = timeline.onUpdate(draw);
    const ro = new ResizeObserver(() => { r.resize(); draw(); });
    ro.observe(canvas.parentElement!);

    return () => {
      unsub();
      ro.disconnect();
      r.dispose();
      rendererRef.current = null;
    };
  }, [timeline, audio, draw]);

  useEffect(draw, [channel, freqScale, draw]);

  useCanvasInteraction(canvasRef, timeline, onSeek);

  return (
    <div className="track" id="spectrogram-track">
      <span className="track-label">Spectrogram</span>
      <div className="scale-toggle">
        <button
          className={`scale-btn${freqScale === 'mel' ? ' active' : ''}`}
          onClick={() => setFreqScale('mel')}
        >
          Mel
        </button>
        <button
          className={`scale-btn${freqScale === 'linear' ? ' active' : ''}`}
          onClick={() => setFreqScale('linear')}
        >
          Linear
        </button>
      </div>
      <canvas ref={canvasRef} />
    </div>
  );
});

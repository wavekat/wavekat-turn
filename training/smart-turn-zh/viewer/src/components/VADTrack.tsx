import { useRef, useEffect, useCallback, memo } from 'react';
import type { Timeline } from '../lib/timeline';
import { VADRenderer } from '../lib/vad';
import { useCanvasInteraction } from '../hooks/useCanvasInteraction';

interface VADTrackProps {
  timeline: Timeline;
  vadBuffer: ArrayBuffer | null;
  entryThreshold: number;
  exitThreshold: number;
  onEntryChange: (v: number) => void;
  onExitChange: (v: number) => void;
  onSeek: (t: number) => void;
}

export const VADTrack = memo(function VADTrack({
  timeline, vadBuffer, entryThreshold, exitThreshold,
  onEntryChange, onExitChange, onSeek,
}: VADTrackProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<VADRenderer | null>(null);

  const entryRef = useRef(entryThreshold);
  const exitRef = useRef(exitThreshold);
  entryRef.current = entryThreshold;
  exitRef.current = exitThreshold;

  const draw = useCallback(() => {
    const r = rendererRef.current;
    if (!r) return;
    r.entryThreshold = entryRef.current;
    r.exitThreshold = exitRef.current;
    r.render();
  }, []);

  // Initialize renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const r = new VADRenderer(canvas, timeline);
    rendererRef.current = r;
    r.resize();

    const unsub = timeline.onUpdate(draw);
    const ro = new ResizeObserver(() => { r.resize(); draw(); });
    ro.observe(canvas.parentElement!);

    return () => { unsub(); ro.disconnect(); rendererRef.current = null; };
  }, [timeline, draw]);

  // Load VAD data when buffer changes
  useEffect(() => {
    if (vadBuffer && rendererRef.current) {
      rendererRef.current.load(vadBuffer);
      draw();
    }
  }, [vadBuffer, draw]);

  // Redraw on threshold changes
  useEffect(draw, [entryThreshold, exitThreshold, draw]);

  useCanvasInteraction(canvasRef, timeline, onSeek);

  return (
    <div className="track" id="vad-track">
      <span className="track-label">VAD</span>
      <canvas ref={canvasRef} />
      <div id="vad-thresholds">
        <label>
          Entry{' '}
          <input
            type="number"
            value={entryThreshold}
            min={0}
            max={1}
            step={0.05}
            onChange={e => onEntryChange(+e.target.value)}
          />
        </label>
        <label>
          Exit{' '}
          <input
            type="number"
            value={exitThreshold}
            min={0}
            max={1}
            step={0.05}
            onChange={e => onExitChange(+e.target.value)}
          />
        </label>
      </div>
    </div>
  );
});

import { useRef, useEffect, useCallback, useMemo, memo } from 'react';
import type { Timeline } from '../lib/timeline';
import { VADRenderer, findVADBlocks, nextVADBlock, prevVADBlock } from '../lib/vad';
import { useCanvasInteraction } from '../hooks/useCanvasInteraction';

interface VADTrackProps {
  timeline: Timeline;
  vadProbs: Float32Array | null;
  entryThreshold: number;
  exitThreshold: number;
  onEntryChange: (v: number) => void;
  onExitChange: (v: number) => void;
  onSeek: (t: number) => void;
}

export const VADTrack = memo(function VADTrack({
  timeline, vadProbs, entryThreshold, exitThreshold,
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

  // Load VAD data when probs change
  useEffect(() => {
    if (vadProbs && rendererRef.current) {
      rendererRef.current.probs = vadProbs;
      draw();
    }
  }, [vadProbs, draw]);

  // Redraw on threshold changes
  useEffect(draw, [entryThreshold, exitThreshold, draw]);

  useCanvasInteraction(canvasRef, timeline, onSeek);

  const blocks = useMemo(
    () => vadProbs ? findVADBlocks(vadProbs, 0.032, entryThreshold, exitThreshold) : [],
    [vadProbs, entryThreshold, exitThreshold],
  );

  const handlePrev = useCallback(() => {
    const b = prevVADBlock(blocks, timeline.cursor);
    if (b) {
      onSeek(b.start);
      const span = timeline.viewEnd - timeline.viewStart;
      timeline.setView(b.start - span * 0.2, b.start - span * 0.2 + span);
    }
  }, [blocks, timeline, onSeek]);

  const handleNext = useCallback(() => {
    const b = nextVADBlock(blocks, timeline.cursor);
    if (b) {
      onSeek(b.start);
      const span = timeline.viewEnd - timeline.viewStart;
      timeline.setView(b.start - span * 0.2, b.start - span * 0.2 + span);
    }
  }, [blocks, timeline, onSeek]);

  return (
    <div className="track" id="vad-track">
      <span className="track-label">VAD</span>
      <canvas ref={canvasRef} />
      <div id="vad-thresholds">
        <div className="vad-nav">
          <button className="vad-nav-btn" onClick={handlePrev} disabled={!vadProbs} title="Previous VAD block ([)">&#9664;</button>
          <button className="vad-nav-btn" onClick={handleNext} disabled={!vadProbs} title="Next VAD block (])">&#9654;</button>
        </div>
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

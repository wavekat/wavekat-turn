import { useRef, useEffect, useCallback, memo } from 'react';
import type { Timeline } from '../lib/timeline';
import type { AudioStore } from '../lib/audio';

interface MinimapProps {
  timeline: Timeline;
  audio: AudioStore;
  channel: number;
}

export const Minimap = memo(function Minimap({ timeline, audio, channel }: MinimapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const channelRef = useRef(channel);
  channelRef.current = channel;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !audio.raw) return;

    const dpr = devicePixelRatio;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    const ctx = canvas.getContext('2d')!;
    const w = rect.width, h = rect.height;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // Coarsest LOD waveform overview
    const ch = channelRef.current;
    const levels = audio.getLOD(ch);
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
    const x1 = (timeline.viewStart / timeline.duration) * w;
    const x2 = (timeline.viewEnd / timeline.duration) * w;
    ctx.fillStyle = 'rgba(79,195,247,0.15)';
    ctx.fillRect(x1, 0, x2 - x1, h);
    ctx.strokeStyle = '#4fc3f7';
    ctx.lineWidth = 1;
    ctx.strokeRect(x1 + 0.5, 0.5, x2 - x1 - 1, h - 1);
  }, [timeline, audio]);

  // Subscribe to timeline updates
  useEffect(() => {
    return timeline.onUpdate(draw);
  }, [timeline, draw]);

  // Redraw on channel change
  useEffect(draw, [channel, draw]);

  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const frac = (e.clientX - rect.left) / rect.width;
    const t = frac * timeline.duration;
    const span = timeline.viewEnd - timeline.viewStart;
    timeline.setView(t - span / 2, t + span / 2);
  }, [timeline]);

  return <canvas ref={canvasRef} id="minimap-canvas" onClick={handleClick} />;
});

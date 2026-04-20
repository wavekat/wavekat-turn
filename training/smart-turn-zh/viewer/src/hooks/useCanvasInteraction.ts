import { useEffect, useRef, type RefObject } from 'react';
import type { Timeline } from '../lib/timeline';

/**
 * Adds wheel zoom, drag pan, and click-to-seek to a canvas element.
 */
export function useCanvasInteraction(
  canvasRef: RefObject<HTMLCanvasElement | null>,
  timeline: Timeline,
  onSeek?: (time: number) => void,
) {
  const onSeekRef = useRef(onSeek);
  onSeekRef.current = onSeek;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.metaKey || e.ctrlKey) {
        const rect = canvas.getBoundingClientRect();
        const frac = (e.clientX - rect.left) / rect.width;
        timeline.zoom(e.deltaY > 0 ? 1.25 : 0.8, frac);
      } else {
        const span = timeline.viewEnd - timeline.viewStart;
        timeline.pan((e.deltaY > 0 ? 0.15 : -0.15) * span);
      }
    };

    let drag = false, startX = 0, startVS = 0, moved = false;

    const onMouseDown = (e: MouseEvent) => {
      drag = true; moved = false; startX = e.clientX; startVS = timeline.viewStart;
    };
    const onMouseMove = (e: MouseEvent) => {
      if (!drag) return;
      const dx = e.clientX - startX;
      if (Math.abs(dx) > 3) moved = true;
      if (!moved) return;
      const rect = canvas.getBoundingClientRect();
      const span = timeline.viewEnd - timeline.viewStart;
      const dt = -(dx / rect.width) * span;
      timeline.setView(startVS + dt, startVS + dt + span);
    };
    const onMouseUp = (e: MouseEvent) => {
      if (!drag) return;
      drag = false;
      if (!moved) {
        const rect = canvas.getBoundingClientRect();
        const t = timeline.xToTime(e.clientX - rect.left, rect.width);
        onSeekRef.current?.(t);
      }
    };

    canvas.addEventListener('wheel', onWheel, { passive: false });
    canvas.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);

    return () => {
      canvas.removeEventListener('wheel', onWheel);
      canvas.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, [canvasRef, timeline]);
}

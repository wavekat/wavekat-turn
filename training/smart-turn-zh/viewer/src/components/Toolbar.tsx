import { useRef, useEffect, useCallback, useState, memo } from 'react';
import type { Timeline } from '../lib/timeline';
import { fmtTime } from '../lib/asr';

const ZOOM_PRESETS = [
  { label: '1s', span: 1 },
  { label: '20s', span: 20 },
  { label: '1m', span: 80 },
  { label: '5m', span: 300 },
  { label: '20m', span: 1200 },
  { label: 'Full', span: 0 },
];

interface ToolbarProps {
  timeline: Timeline;
  onFilesSelected: (files: File[]) => void;
  wavStatus: string;
  vadStatus: string;
  asrStatus: string;
  channelCount: number;
  channel: number;
  onChannelChange: (ch: number) => void;
  playing: boolean;
  canPlay: boolean;
  onPlayToggle: () => void;
  gainValue: number;
  onGainChange: (v: number) => void;
}

export const Toolbar = memo(function Toolbar({
  timeline, onFilesSelected,
  wavStatus, vadStatus, asrStatus,
  channelCount, channel, onChannelChange,
  playing, canPlay, onPlayToggle,
  gainValue, onGainChange,
}: ToolbarProps) {
  const timeRef = useRef<HTMLSpanElement>(null);
  const [activeZoom, setActiveZoom] = useState<number | null>(null);

  // Update time display imperatively (60fps during playback)
  useEffect(() => {
    const update = () => {
      if (timeRef.current) {
        timeRef.current.textContent = `${fmtTime(timeline.cursor)} / ${fmtTime(timeline.duration)}`;
      }
    };
    update();
    return timeline.onUpdate(update);
  }, [timeline]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (files.length) onFilesSelected(files);
  }, [onFilesSelected]);

  const handleZoom = useCallback((span: number, idx: number) => {
    if (span === 0) {
      timeline.setView(0, timeline.duration);
    } else {
      const center = (timeline.viewStart + timeline.viewEnd) / 2;
      timeline.setView(center - span / 2, center + span / 2);
    }
    setActiveZoom(idx);
  }, [timeline]);

  return (
    <header id="toolbar">
      <div id="file-controls">
        <label className="file-btn">
          Open files
          <input type="file" multiple hidden onChange={handleFileInput} />
        </label>
        <span className="status">{wavStatus}</span>
        <span className="status">{vadStatus}</span>
        <span className="status">{asrStatus}</span>
      </div>
      <div id="zoom-controls">
        <button
          className="zoom-btn"
          onClick={() => { timeline.zoom(0.5, 0.5); setActiveZoom(null); }}
          title="Zoom in (+)"
        >
          +
        </button>
        <button
          className="zoom-btn"
          onClick={() => { timeline.zoom(2, 0.5); setActiveZoom(null); }}
          title="Zoom out (-)"
        >
          &minus;
        </button>
        <span className="zoom-sep" />
        {ZOOM_PRESETS.map((z, i) => (
          <button
            key={z.label}
            className={`zoom-btn${activeZoom === i ? ' active' : ''}`}
            onClick={() => handleZoom(z.span, i)}
          >
            {z.label}
          </button>
        ))}
      </div>
      <div id="playback-controls">
        <select
          id="channel-select"
          value={channel}
          onChange={e => onChannelChange(+e.target.value)}
          disabled={channelCount === 0}
        >
          <option value={-1}>All</option>
          {Array.from({ length: channelCount }, (_, i) => (
            <option key={i} value={i}>Ch {i + 1}</option>
          ))}
        </select>
        <button
          id="play-btn"
          disabled={!canPlay}
          onClick={onPlayToggle}
          title="Space"
          dangerouslySetInnerHTML={{ __html: playing ? '&#9646;&#9646;' : '&#9654;' }}
        />
        <label id="gain-label">
          Vol{' '}
          <input
            type="range"
            id="gain-slider"
            min={0}
            max={20}
            step={0.1}
            value={gainValue}
            onChange={e => onGainChange(+e.target.value)}
          />
          <span id="gain-value">{gainValue.toFixed(1)}x</span>
        </label>
        <span ref={timeRef} id="time-display">--:-- / --:--</span>
      </div>
    </header>
  );
});

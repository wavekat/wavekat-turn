import { useState, useCallback, useRef, useEffect } from 'react';
import type { Timeline } from '../lib/timeline';
import type { AudioStore } from '../lib/audio';

export function usePlayback(timeline: Timeline, audio: AudioStore) {
  const [playing, setPlaying] = useState(false);

  const actxRef = useRef<AudioContext | null>(null);
  const srcRef = useRef<AudioBufferSourceNode | null>(null);
  const gainNodeRef = useRef<GainNode | null>(null);
  const playBufRef = useRef<AudioBuffer | null>(null);
  const playChannelRef = useRef(-1);
  const playT0Ref = useRef(0);
  const playOffRef = useRef(0);
  const gainValueRef = useRef(10.0);
  const playingRef = useRef(false);
  const rafIdRef = useRef(0);

  const stop = useCallback(() => {
    playingRef.current = false;
    cancelAnimationFrame(rafIdRef.current);

    if (srcRef.current) {
      srcRef.current.onended = null;
      try { srcRef.current.stop(); } catch { /* already stopped */ }
      srcRef.current = null;
    }
    if (actxRef.current) {
      const elapsed = actxRef.current.currentTime - playT0Ref.current;
      timeline.cursor = Math.min(playOffRef.current + elapsed, timeline.duration);
      actxRef.current.close();
      actxRef.current = null;
    }
    gainNodeRef.current = null;
    setPlaying(false);
    timeline.flush();
  }, [timeline]);

  const stopRef = useRef(stop);
  stopRef.current = stop;

  const play = useCallback((offset: number, channel: number) => {
    if (!audio.raw) return;

    // Stop any existing playback
    if (playingRef.current) stopRef.current();

    // Ensure buffer for channel
    if (!playBufRef.current || playChannelRef.current !== channel) {
      playBufRef.current = audio.createAudioBuffer(channel);
      playChannelRef.current = channel;
    }
    const buf = playBufRef.current;
    if (!buf) return;

    const actx = new AudioContext({ sampleRate: audio.sampleRate });
    const gainNode = actx.createGain();
    gainNode.gain.value = gainValueRef.current;
    gainNode.connect(actx.destination);
    const src = actx.createBufferSource();
    src.buffer = buf;
    src.connect(gainNode);
    src.start(0, offset);
    src.onended = () => stopRef.current();

    actxRef.current = actx;
    srcRef.current = src;
    gainNodeRef.current = gainNode;
    playT0Ref.current = actx.currentTime;
    playOffRef.current = offset;
    playingRef.current = true;
    setPlaying(true);

    const loop = () => {
      if (!playingRef.current) return;
      const a = actxRef.current;
      if (!a) return;
      const t = playOffRef.current + (a.currentTime - playT0Ref.current);
      if (t >= timeline.duration) { stopRef.current(); return; }
      timeline.cursor = t;
      // Auto-scroll when cursor exits viewport
      if (t > timeline.viewEnd) {
        const span = timeline.viewEnd - timeline.viewStart;
        timeline.viewStart = t;
        timeline.viewEnd = t + span;
      }
      timeline.flush();
      rafIdRef.current = requestAnimationFrame(loop);
    };
    rafIdRef.current = requestAnimationFrame(loop);
  }, [audio, timeline]);

  const setGain = useCallback((value: number) => {
    gainValueRef.current = value;
    if (gainNodeRef.current) gainNodeRef.current.gain.value = value;
  }, []);

  const invalidateBuffer = useCallback(() => {
    playBufRef.current = null;
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      playingRef.current = false;
      cancelAnimationFrame(rafIdRef.current);
      if (srcRef.current) {
        srcRef.current.onended = null;
        try { srcRef.current.stop(); } catch { /* already stopped */ }
      }
      actxRef.current?.close();
    };
  }, []);

  return { playing, play, stop, setGain, invalidateBuffer };
}

import { useRef, useEffect, useCallback, memo } from 'react';
import type { Timeline } from '../lib/timeline';
import {
  type Sentence,
  matchHlSet, fmtTime, fmtMs, zoomToSentence,
} from '../lib/asr';

interface ASRPanelProps {
  timeline: Timeline;
  sentences: Sentence[];
  searchQuery: string;
  searchResults: number[];
  searchResultIdx: number;
  onSearchChange: (q: string) => void;
  onNext: () => void;
  onPrev: () => void;
  onSeek: (time: number) => void;
  playing: boolean;
}

export const ASRPanel = memo(function ASRPanel({
  timeline, sentences, searchQuery, searchResults, searchResultIdx,
  onSearchChange, onNext, onPrev, onSeek, playing,
}: ASRPanelProps) {
  const listRef = useRef<HTMLDivElement>(null);
  const activeCharRef = useRef<HTMLElement | null>(null);
  const activeRangeRef = useRef<[number, number]>([-1, -1]);

  // Keep sentences in ref for stable callbacks
  const sentencesRef = useRef(sentences);
  sentencesRef.current = sentences;

  const clearHighlight = useCallback(() => {
    if (activeCharRef.current) {
      activeCharRef.current.classList.remove('char-active');
      activeCharRef.current = null;
      activeRangeRef.current = [-1, -1];
    }
  }, []);

  const highlightAt = useCallback((timeMs: number) => {
    // Skip if still within current character
    if (
      activeCharRef.current &&
      timeMs >= activeRangeRef.current[0] &&
      timeMs < activeRangeRef.current[1]
    ) {
      return;
    }
    clearHighlight();

    const sents = sentencesRef.current;
    let si = -1;
    for (let i = 0; i < sents.length; i++) {
      if (timeMs >= sents[i].start && timeMs <= sents[i].end) { si = i; break; }
    }
    if (si < 0) return;

    const s = sents[si];
    let charIdx = -1;
    for (let ci = 0; ci < s.chars.length; ci++) {
      const c = s.chars[ci];
      if (c.start >= 0 && timeMs >= c.start && timeMs < c.end) { charIdx = ci; break; }
    }
    if (charIdx < 0) return;

    const sentEl = listRef.current?.querySelector(`[data-idx="${si}"]`);
    if (!sentEl) return;
    const txtEl = sentEl.querySelector('.txt');
    if (!txtEl) return;
    const el = txtEl.children[charIdx] as HTMLElement | undefined;
    if (!el) return;

    el.classList.add('char-active');
    activeCharRef.current = el;
    activeRangeRef.current = [s.chars[charIdx].start, s.chars[charIdx].end];
    sentEl.scrollIntoView?.({ block: 'nearest', behavior: 'auto' });
  }, [clearHighlight]);

  // Subscribe to timeline for karaoke highlighting
  useEffect(() => {
    return timeline.onUpdate(() => highlightAt(timeline.cursor * 1000));
  }, [timeline, highlightAt]);

  // Re-apply karaoke after DOM rebuild
  useEffect(() => {
    clearHighlight();
    highlightAt(timeline.cursor * 1000);
  }, [sentences, searchQuery, searchResults, searchResultIdx, clearHighlight, highlightAt, timeline]);

  // Scroll current search result into view
  const currentSent = searchResultIdx >= 0 ? searchResults[searchResultIdx] : -1;
  useEffect(() => {
    if (currentSent >= 0) {
      const el = listRef.current?.querySelector(`[data-idx="${currentSent}"]`);
      el?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  }, [currentSent]);

  // Handle click on a sentence or character
  const handleClick = useCallback((e: React.MouseEvent) => {
    const el = (e.target as HTMLElement).closest('.sentence') as HTMLElement | null;
    if (!el) return;
    const sentIdx = +el.dataset.idx!;

    const charEl = (e.target as HTMLElement).closest('.char') as HTMLElement | null;
    const t = charEl ? +charEl.dataset.cs! / 1000 : +el.dataset.start! / 1000;

    zoomToSentence(timeline, sentencesRef.current, sentIdx);
    onSeek(t);
  }, [timeline, onSeek]);

  const matchSet = new Set(searchResults);
  const searchCount = searchQuery
    ? (searchResults.length ? `${searchResultIdx + 1}/${searchResults.length}` : 'No results')
    : '';

  return (
    <div id="asr-panel">
      <div id="search-bar">
        <input
          type="text"
          id="search-input"
          placeholder="Search ASR text... (F to focus)"
          value={searchQuery}
          onChange={e => onSearchChange(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter') {
              e.preventDefault();
              e.shiftKey ? onPrev() : onNext();
            }
          }}
        />
        <span id="search-count">{searchCount}</span>
        <button
          disabled={!searchResults.length}
          onClick={onPrev}
          title="Shift+Enter"
          dangerouslySetInnerHTML={{ __html: '&#9664;' }}
        />
        <button
          disabled={!searchResults.length}
          onClick={onNext}
          title="Enter"
          dangerouslySetInnerHTML={{ __html: '&#9654;' }}
        />
      </div>
      <div ref={listRef} id="transcript-list" onClick={handleClick}>
        {sentences.map((s, i) => {
          const cls = i === currentSent
            ? 'sentence current'
            : matchSet.has(i) ? 'sentence match' : 'sentence';
          const hlSet = searchQuery && matchSet.has(i)
            ? matchHlSet(s.text, searchQuery) : null;

          return (
            <div key={i} className={cls} data-idx={i} data-start={s.start} data-end={s.end}>
              <span className="time">{fmtTime(s.start / 1000)}</span>
              <span className="txt">
                {s.chars.map((c, ci) => {
                  const hl = hlSet?.has(ci) ? ' search-hl' : '';
                  if (c.start >= 0) {
                    return (
                      <span
                        key={ci}
                        className={`char${hl}`}
                        data-cs={c.start}
                        data-ce={c.end}
                        title={`${fmtMs(c.start)} \u2192 ${fmtMs(c.end)}`}
                      >
                        {c.char}
                      </span>
                    );
                  }
                  return <span key={ci} className={`punc${hl}`}>{c.char}</span>;
                })}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
});

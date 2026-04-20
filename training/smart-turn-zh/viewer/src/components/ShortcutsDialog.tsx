import { memo } from 'react';

const SHORTCUTS = [
  { key: 'Space', desc: 'Play / pause (or replay loop)' },
  { key: 'F', desc: 'Focus search' },
  { key: 'Escape', desc: 'Clear loop / blur input' },
  { key: '+ / −', desc: 'Zoom in / out' },
  { key: '← / →', desc: 'Pan left / right' },
  { key: '[ / ]', desc: 'Previous / next VAD block' },
  { key: 'Shift + drag', desc: 'Select loop range' },
  { key: '0', desc: 'All channels' },
  { key: '1–9', desc: 'Select channel' },
  { key: '?', desc: 'Toggle this dialog' },
];

export const ShortcutsDialog = memo(function ShortcutsDialog({ onClose }: { onClose: () => void }) {
  return (
    <div className="shortcuts-overlay" onClick={onClose}>
      <div className="shortcuts-dialog" onClick={e => e.stopPropagation()}>
        <div className="shortcuts-header">
          <span>Keyboard Shortcuts</span>
          <button className="shortcuts-close" onClick={onClose}>&times;</button>
        </div>
        <div className="shortcuts-body">
          {SHORTCUTS.map(s => (
            <div key={s.key} className="shortcut-row">
              <kbd>{s.key}</kbd>
              <span>{s.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
});

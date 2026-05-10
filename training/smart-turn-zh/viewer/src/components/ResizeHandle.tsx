import { memo } from 'react';

interface ResizeHandleProps {
  direction: 'row' | 'col';
  onDrag: (delta: number) => void;
}

export const ResizeHandle = memo(function ResizeHandle({
  direction,
  onDrag,
}: ResizeHandleProps) {
  const onMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    let last = direction === 'col' ? e.clientX : e.clientY;

    const move = (ev: MouseEvent) => {
      const cur = direction === 'col' ? ev.clientX : ev.clientY;
      onDrag(cur - last);
      last = cur;
    };
    const up = () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      document.body.style.removeProperty('cursor');
      document.body.style.removeProperty('user-select');
    };

    document.body.style.cursor =
      direction === 'col' ? 'col-resize' : 'row-resize';
    document.body.style.userSelect = 'none';
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  };

  return (
    <div
      className={`resize-handle resize-${direction}`}
      onMouseDown={onMouseDown}
    />
  );
});

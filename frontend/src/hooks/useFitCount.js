import { useEffect, useRef, useState } from 'react';

/**
 * How many items fit in a container, recomputed whenever it is resized.
 *
 * `measure(el)` receives the container and returns the count (or null when
 * it cannot tell yet). Hidden containers (0×0, e.g. an inactive tab) keep
 * the last value, so switching tabs does not reset the page size.
 *
 * Returns [ref, count] — attach `ref` to the container.
 */
export function useFitCount(measure, fallback, deps = []) {
  const ref = useRef(null);
  const [count, setCount] = useState(fallback);
  const measureRef = useRef(measure);
  measureRef.current = measure;

  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return undefined;
    const update = () => {
      if (!el.clientWidth || !el.clientHeight) return;
      const n = measureRef.current(el);
      if (!n) return;
      // Objects (e.g. {cols, rows}) compare by value so equal results do not re-render.
      setCount((prev) => (JSON.stringify(prev) === JSON.stringify(n) ? prev : n));
    };
    const ro = new ResizeObserver(update);
    ro.observe(el);
    update();
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return [ref, count];
}

/**
 * Measure for a scrolling table wrapper: rows that fit under the sticky
 * header, using the height of the first rendered body row.
 */
export function fitTableRows(defaultRowHeight, min = 8) {
  return (wrap) => {
    const head = wrap.querySelector('thead');
    const row = wrap.querySelector('tbody tr');
    const rowH = row?.getBoundingClientRect().height || defaultRowHeight;
    const headH = head?.getBoundingClientRect().height || 0;
    // A horizontal scrollbar takes ~10px from the visible area.
    const scrollbar = wrap.scrollWidth > wrap.clientWidth ? 10 : 0;
    return Math.max(min, Math.floor((wrap.clientHeight - headH - scrollbar) / rowH));
  };
}

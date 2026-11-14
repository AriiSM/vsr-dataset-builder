import { useState, useEffect } from 'react';

/**
 * Returns the value "delayed" by `delayMs` — useful for search fields:
 * the filter is applied only after the user has stopped typing,
 * not on every keystroke.
 */
export function useDebouncedValue(value, delayMs = 150) {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);

  return debounced;
}

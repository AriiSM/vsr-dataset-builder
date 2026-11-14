import { useState, useEffect } from 'react';

/**
 * Like `useState`, but the value survives a page reload,
 * being saved in localStorage under the given key.
 *
 *   const [tab, setTab] = useLocalStorage('vsr-active-tab', 'process');
 */
export function useLocalStorage(key, initialValue) {
  const [value, setValue] = useState(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored !== null ? JSON.parse(stored) : initialValue;
    } catch {
      return initialValue;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // localStorage unavailable (e.g. private mode) — ignore, value stays in memory
    }
  }, [key, value]);

  return [value, setValue];
}

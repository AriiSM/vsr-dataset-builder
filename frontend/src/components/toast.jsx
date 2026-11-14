import { useState, useEffect } from 'react';

/**
 * Tiny toast notification system — replaces the blocking alert()/confirm()
 * browser dialogs so the keyboard-driven flow (especially in Review) is
 * never interrupted.
 *
 * Usage from anywhere (no context/provider needed):
 *   toast.success('Saved');
 *   toast.error('Save failed: ...');
 *   toast.info('Reloading queue');
 *   toast.confirm('Stop the running pipeline?', () => api.stopPipeline());
 *
 * The <Toaster /> component (mounted once in App) listens to a module-level
 * subscriber list and renders the stack in the bottom-right corner.
 */

let nextToastId = 1;
// The single active listener (the mounted <Toaster />). A list would allow
// several, but the app mounts exactly one.
let listener = null;

/** Pushes a toast to the mounted Toaster; no-op if none is mounted yet. */
function push(entry) {
  listener?.({ id: nextToastId++, ...entry });
}

export const toast = {
  success: (message) => push({ type: 'success', message, timeoutMs: 3000 }),
  info:    (message) => push({ type: 'info',    message, timeoutMs: 3000 }),
  error:   (message) => push({ type: 'error',   message, timeoutMs: 6000 }),
  /** Toast with Confirm/Cancel buttons; stays until the user chooses. */
  confirm: (message, onConfirm) =>
    push({ type: 'confirm', message, onConfirm, timeoutMs: null }),
};

/** Renders the toast stack. Mount exactly once, at the app root. */
export function Toaster() {
  const [toasts, setToasts] = useState([]);

  useEffect(() => {
    listener = (entry) => {
      setToasts((prev) => [...prev, entry]);
      if (entry.timeoutMs) {
        setTimeout(() => dismiss(entry.id), entry.timeoutMs);
      }
    };
    return () => { listener = null; };
  }, []);

  function dismiss(id) {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }

  return (
    <div className="toast-stack" role="status" aria-live="polite">
      {toasts.map((t) => (
        <div key={t.id} className={`toast toast-${t.type}`}>
          <span className="toast-msg">{t.message}</span>
          {t.type === 'confirm' ? (
            <span className="toast-actions">
              <button
                className="btn-micro toast-confirm-btn"
                onClick={() => { dismiss(t.id); t.onConfirm?.(); }}
              >
                Confirm
              </button>
              <button className="btn-micro" onClick={() => dismiss(t.id)}>Cancel</button>
            </span>
          ) : (
            <button className="toast-close" aria-label="Dismiss notification" onClick={() => dismiss(t.id)}>×</button>
          )}
        </div>
      ))}
    </div>
  );
}

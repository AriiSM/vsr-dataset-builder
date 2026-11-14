import { useEffect, useState } from 'react';
import { api } from '../api.js';

/**
 * Live backend-status badge for the header.
 *
 * Polls /api/health and shows one of four states:
 *   · checking — first request still in flight (neutral gray)
 *   · online   — API answered and the catalog DB exists (green)
 *   · no-db    — API answered but dataset.db is missing (amber):
 *                the server runs, but stats/review would be empty
 *   · offline  — request failed / backend not running (red)
 *
 * The legacy Flask server has no /api/health, so on a failed health check
 * we fall back to /api/status before declaring the backend offline —
 * the badge then still works during the Flask→FastAPI transition.
 */

const POLL_ONLINE_MS = 15000; // relaxed while everything is fine
const POLL_OFFLINE_MS = 5000; // faster, so recovery shows quickly

const STATE_META = {
  checking: { label: 'API …', title: 'Checking backend availability…' },
  online:   { label: 'API online', title: '' }, // title filled from health data
  'no-db':  { label: 'API · no DB', title: 'Backend running, but dataset.db was not found — stats and review will be empty.' },
  offline:  { label: 'API offline', title: 'Backend unreachable. Start it with: python backend/run_api.py' },
};

export function BackendStatus() {
  const [state, setState] = useState('checking');
  const [detail, setDetail] = useState('');

  useEffect(() => {
    let cancelled = false;
    let timer = null;

    const check = async () => {
      let next = 'offline';
      let nextDetail = '';
      try {
        const health = await api.getHealth();
        next = health.catalog_db_exists === false ? 'no-db' : 'online';
        nextDetail = [
          'Backend online',
          health.git_sha ? `build ${String(health.git_sha).slice(0, 7)}` : null,
          health.catalog_db_exists === false ? 'dataset.db MISSING' : 'dataset.db found',
        ].filter(Boolean).join(' · ');
      } catch {
        // Legacy Flask fallback: no /api/health there, but /api/status exists.
        try {
          await api.getStatus();
          next = 'online';
          nextDetail = 'Backend online (legacy server without /api/health)';
        } catch { /* truly unreachable */ }
      }
      if (cancelled) return;
      setState(next);
      setDetail(nextDetail);
      timer = setTimeout(check, next === 'offline' ? POLL_OFFLINE_MS : POLL_ONLINE_MS);
    };

    check();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  const meta = STATE_META[state];
  return (
    <span className={`backend-status is-${state}`} title={detail || meta.title}>
      <span className="backend-dot" />
      {meta.label}
    </span>
  );
}

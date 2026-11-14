import { VideoPicker } from './VideoPicker.jsx';

/**
 * The run-mode selector + the forms specific to each mode.
 * A "controlled" component: all state comes from ProcessTab via props.
 */

/**
 * The available modes, in workflow order: import links → process them →
 * resume/retry when something breaks. ("Single video" and a separate
 * "Batch failed" mode were dropped — retrying failed videos is a checkbox
 * on Resume instead.)
 */
const MODES = [
  { value: 'bulk-import',   title: 'Bulk import', subtitle: 'Download many URLs', wide: true },
  { value: 'batch-pending', title: 'Batch',       subtitle: 'Process pending' },
  { value: 'resume',        title: 'Resume',      subtitle: 'Continue / retry' },
];

/**
 * The short explanation of each mode, displayed below the selector.
 * Written from the user's side of the screen: what pressing Start will do,
 * in the app's own vocabulary (video registry, source videos, clips,
 * dataset segments) — no file names or status codes.
 */
const MODE_HINTS = {
  'bulk-import': (
    <>Paste YouTube links below — each video is <strong>downloaded and added to the
      video registry</strong>, ready for processing. Nothing is processed yet:
      run <strong>Batch</strong> afterwards.</>
  ),
  'batch-pending': (
    <>Processes every downloaded video that hasn't been processed yet: cuts each
      source video into clips, filters them, and exports the good ones as{' '}
      <strong>dataset segments</strong>. This is the main run — start it after a
      bulk import.</>
  ),
  resume: (
    <>Picks up videos that were <strong>interrupted mid-processing</strong> and
      continues them from where they stopped — nothing already done is redone.
      Check <strong>Retry failed videos from scratch</strong> to instead re-run
      every failed video from the beginning.</>
  ),
};

export function ModeSelector({
  mode, onModeChange, form, onFormChange,
  pickerVideos, selectedIds, onSelectionChange,
}) {
  const isBulk = mode === 'bulk-import';
  const isBatch = !isBulk;

  /** Updates a single form field. */
  const set = (field) => (event) => {
    const el = event.target;
    onFormChange({ ...form, [field]: el.type === 'checkbox' ? el.checked : el.value });
  };

  return (
    <>
      {/* ── Mode selection ── */}
      <div className="machine-section">
        <div className="machine-section-hdr">
          <span className="machine-label">Run mode</span>
        </div>
        <div className="mode-grid">
          {MODES.map((m) => (
            <label key={m.value} className={`mode-opt ${m.wide ? 'mode-opt-wide' : ''}`}>
              <input
                type="radio"
                name="run_mode"
                value={m.value}
                checked={mode === m.value}
                onChange={() => onModeChange(m.value)}
              />
              <span>
                {m.title}
                <em>{m.subtitle}</em>
              </span>
            </label>
          ))}
        </div>
        <div className="mode-hint">{MODE_HINTS[mode]}</div>
      </div>

      {/* ── Fields for the "bulk-import" mode ── */}
      {isBulk && (
        <div className="machine-section">
          <div className="machine-section-hdr">
            <span className="machine-label">Bulk import</span>
          </div>
          <div className="option-row">
            <span className="option-label">Prefix</span>
            <input
              className="text-input-sm"
              type="text"
              placeholder="e.g. md"
              autoComplete="off"
              value={form.bulkPrefix}
              onChange={set('bulkPrefix')}
            />
          </div>
          <div className="option-row" style={{ marginTop: 6 }}>
            <span className="option-label">Region</span>
            <select className="select-sm" value={form.bulkRegion} onChange={set('bulkRegion')}>
              <option value="UNKNOWN">Unknown</option>
              <option value="RO">RO · Romania</option>
              <option value="MD">MD · Moldova</option>
              <option value="DIASPORA">Diaspora</option>
            </select>
          </div>
          <div className="option-row" style={{ marginTop: 6 }}>
            <span className="option-label">Source</span>
            <select className="select-sm" value={form.bulkSource} onChange={set('bulkSource')}>
              <option value="YouTube_CC">YouTube_CC</option>
              <option value="TEDx">TEDx</option>
              <option value="Interview">Interview</option>
              <option value="Lecture">Lecture</option>
              <option value="Podcast">Podcast</option>
              <option value="News">News</option>
              <option value="Other">Other</option>
            </select>
          </div>
          <label className="rv-opt-label" style={{ marginTop: 8 }}>
            <input
              type="checkbox"
              checked={form.bulkPreDownloaded}
              onChange={set('bulkPreDownloaded')}
            />
            Videos already downloaded (in data/raw) — map them, fetch metadata only
          </label>
          <div style={{ marginTop: 8 }}>
            <span
              className="option-label"
              style={{ display: 'block', marginBottom: 4, width: 'auto' }}
            >
              {form.bulkPreDownloaded
                ? 'video_id + URL (one pair per line)'
                : 'YouTube URLs (one per line)'}
            </span>
            <textarea
              className="text-input"
              rows={6}
              placeholder={
                form.bulkPreDownloaded
                  ? 'md_001 https://youtube.com/watch?v=...\nmd_002 https://youtu.be/...\n# id-ul trebuie să corespundă cu data/raw/{id}.mp4'
                  : 'https://youtube.com/watch?v=...\nhttps://youtu.be/...\n# lines starting with # are ignored'
              }
              style={{ resize: 'vertical', fontFamily: 'var(--font-mono)', fontSize: 11, width: '100%' }}
              value={form.bulkUrls}
              onChange={set('bulkUrls')}
            />
          </div>
          <label className="rv-opt-label" style={{ marginTop: 8 }}>
            <input type="checkbox" checked={form.bulkNoCcCheck} onChange={set('bulkNoCcCheck')} />
            Skip Creative Commons check
          </label>
        </div>
      )}

      {/* ── Options for the batch / resume modes ── */}
      {isBatch && (
        <div className="machine-section">
          <div className="machine-section-hdr">
            <span className="machine-label">Options</span>
          </div>
          <div className="option-row">
            <span className="option-label">Limit</span>
            <input
              className="text-input-sm"
              type="number"
              min={1}
              placeholder="all"
              value={form.limit}
              onChange={set('limit')}
            />
          </div>
          {/* Resume's one decision: continue interrupted (default) or
              re-run every failed video from the beginning. */}
          {mode === 'resume' && (
            <label className="rv-opt-label" style={{ margin: '8px 0 2px' }}>
              <input
                type="checkbox"
                checked={form.retryFailed}
                onChange={set('retryFailed')}
              />
              Retry failed videos from scratch
            </label>
          )}
        </div>
      )}

      {/* ── Which videos will run (checkbox picker) ── */}
      {isBatch && (
        <div className="machine-section">
          <div className="machine-section-hdr">
            <span className="machine-label">Videos to process</span>
          </div>
          <VideoPicker
            videos={pickerVideos}
            selectedIds={selectedIds}
            onSelectionChange={onSelectionChange}
          />
        </div>
      )}

      {/* ── YouTube cookies (advanced, collapsible) ── */}
      <details className="machine-section-details">
        <summary className="machine-section-summary">YouTube cookies</summary>
        <div className="machine-section-body">
          <div className="option-row" style={{ marginBottom: 6 }}>
            <span className="option-label">File</span>
            <input
              className="text-input-sm"
              type="text"
              placeholder="./cookies.txt"
              value={form.cookies}
              onChange={set('cookies')}
            />
          </div>
          <div className="option-row">
            <span className="option-label">Browser</span>
            <select className="select-sm" value={form.cookiesBrowser} onChange={set('cookiesBrowser')}>
              <option value="">— none —</option>
              <option value="firefox">firefox</option>
              <option value="chrome">chrome</option>
              <option value="edge">edge</option>
              <option value="brave">brave</option>
            </select>
          </div>
        </div>
      </details>
    </>
  );
}

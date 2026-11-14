import { useState, useMemo } from 'react';
import { useFitCount, fitTableRows } from '../../hooks/useFitCount.js';
import { formatDurationShort } from '../../utils/format.js';
import { useDebouncedValue } from '../../hooks/useDebouncedValue.js';
import { Pagination } from '../../components/Pagination.jsx';
import { VideoDocumentModal } from './VideoDocumentModal.jsx';

const VIDEOS_PER_PAGE = 25; // until the table has been measured

/** The status filters available above the table. */
const STATUS_FILTERS = [
  { value: 'all',       label: 'All' },
  { value: 'completed', label: 'Done' },
  { value: 'validated', label: 'Validated' },
  { value: 'pending',   label: 'Pending' },
  { value: 'failed',    label: 'Failed' },
];

/**
 * Columns hidden when every video leaves them empty (manual metadata that
 * was never filled in). Numeric ones also count 0 as empty — the table
 * already shows "—" for them.
 */
const OPTIONAL_COLUMNS = [
  'region', 'source', 'source_channel', 'license',
  'avg_asd_score', 'avg_syncnet_conf',
  'gender', 'age_group', 'environment', 'background_noise', 'num_speakers',
  'processed_date', 'error_message',
];
const NUMERIC_COLUMNS = new Set(['avg_asd_score', 'avg_syncnet_conf', 'num_speakers']);
/** Always-shown columns: ID, YT, Title, Status, Source, Extracted, Mined%, Segs. */
const FIXED_COLUMN_COUNT = 8;

function isFilled(key, value) {
  if (value == null) return false;
  const text = String(value).trim();
  if (text === '' || text === '—') return false;
  return !(NUMERIC_COLUMNS.has(key) && Number(text) === 0);
}

/** Simple cell: displays the value, or "—" if it's missing. */
function Cell({ value, numeric = false }) {
  const isEmpty = value === '' || value == null || value === '—';
  return (
    <td className="dim" style={numeric ? { fontVariantNumeric: 'tabular-nums' } : undefined}>
      {isEmpty ? <span className="dim">—</span> : String(value)}
    </td>
  );
}

/**
 * The video registry: all the columns from videos_master.csv, with search
 * by ID/title, a status filter, and pagination. The table is wide (up to 21
 * columns; all-empty ones are hidden) → horizontal scrolling inside the section.
 */
export function VideosTable({ videos, headerActions = null }) {
  const [statusFilter, setStatusFilter] = useState('all');
  const [searchInput, setSearchInput] = useState('');
  const [page, setPage] = useState(1);
  const [openVideo, setOpenVideo] = useState(null);
  const search = useDebouncedValue(searchInput, 120);

  // Filtering is recomputed only when the data or the criteria change.
  const filtered = useMemo(() => {
    let arr = videos;
    if (statusFilter !== 'all') arr = arr.filter((v) => v.status === statusFilter);
    const q = search.trim().toLowerCase();
    if (q) {
      arr = arr.filter(
        (v) =>
          (v.video_id || '').toLowerCase().includes(q) ||
          (v.title || '').toLowerCase().includes(q)
      );
    }
    return arr;
  }, [videos, statusFilter, search]);

  // Which optional columns have data in at least one video (over all videos,
  // so columns don't jump around while filtering).
  const shown = useMemo(() => {
    const filled = new Set();
    for (const key of OPTIONAL_COLUMNS) {
      if (videos.some((v) => isFilled(key, v[key]))) filled.add(key);
    }
    return filled;
  }, [videos]);
  const columnCount = FIXED_COLUMN_COUNT + shown.size;

  // Rows per page = rows that fit on screen (no empty band, no scroll).
  const [tableWrapRef, perPage] = useFitCount(fitTableRows(34), VIDEOS_PER_PAGE, [filtered.length > 0]);
  const totalPages = Math.max(1, Math.ceil(filtered.length / perPage));
  const safePage = Math.min(page, totalPages);
  const pageRows = filtered.slice((safePage - 1) * perPage, safePage * perPage);

  return (
    <div className="browse-section browse-section-videos">
      <div className="cmd-panel-hdr">
        <span className="cmd-panel-label">Video registry</span>
        <span className="cmd-panel-count">
          {filtered.length ? `${filtered.length.toLocaleString()} shown` : ''}
        </span>
        <div className="cmd-right-actions">
          {headerActions}
          <input
            type="text"
            className="vocab-search-input"
            placeholder="Search ID or title..."
            style={{ width: 200, flexShrink: 0 }}
            value={searchInput}
            onChange={(e) => { setSearchInput(e.target.value); setPage(1); }}
          />
          <div className="cmd-filter-group">
            {STATUS_FILTERS.map((f) => (
              <button
                key={f.value}
                className={`btn-micro stats-filter ${statusFilter === f.value ? 'active' : ''}`}
                onClick={() => { setStatusFilter(f.value); setPage(1); }}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="cmd-table-wrap browse-table-scroll" ref={tableWrapRef}>
        <table className="videos-table">
          <thead>
            <tr>
              <th>Video ID</th>
              <th title="Open on YouTube">YT</th>
              <th>Title</th>
              {shown.has('region') && <th>Region</th>}
              <th>Status</th>
              {shown.has('source') && <th title="Source type (Interview / TEDx / Lecture / ...)">Type</th>}
              {shown.has('source_channel') && <th>Channel</th>}
              {shown.has('license') && <th>License</th>}
              <th title="Original source-video duration">Source</th>
              <th title="Total duration of exported segments">Extracted</th>
              <th title="Extracted ÷ Source — fraction of raw video that became dataset material">Mined%</th>
              <th title="Segment count">Segs</th>
              {shown.has('avg_asd_score') && <th title="Average Active Speaker Detection score">ASD</th>}
              {shown.has('avg_syncnet_conf') && <th title="Average SyncNet audio-visual sync confidence">SyncNet</th>}
              {shown.has('gender') && <th title="Speaker gender">Gen</th>}
              {shown.has('age_group') && <th title="Speaker age group">Age</th>}
              {shown.has('environment') && <th title="Recording environment">Env</th>}
              {shown.has('background_noise') && <th title="Background noise level">Noise</th>}
              {shown.has('num_speakers') && <th title="Number of distinct speakers">#Spk</th>}
              {shown.has('processed_date') && <th>Processed</th>}
              {shown.has('error_message') && <th title="Pipeline error message, if processing failed">Error</th>}
            </tr>
          </thead>
          <tbody>
            {pageRows.length === 0 ? (
              <tr>
                <td colSpan={columnCount}style={{ textAlign: 'center', padding: 24, color: 'var(--text-muted)' }}>
                  {videos.length === 0 ? 'No videos in Excel yet.' : 'No videos match filter.'}
                </td>
              </tr>
            ) : (
              pageRows.map((v) => {
                const status = v.status || 'unknown';
                const srcS = parseFloat(v.duration_seconds);
                const extS = parseFloat(v.total_duration_extracted);
                const title = String(v.title || '').trim();

                // The "mining" ratio — colored by how productive the video was.
                let mined = <span className="dim">—</span>;
                if (Number.isFinite(srcS) && srcS > 0 && Number.isFinite(extS)) {
                  const pct = (extS / srcS) * 100;
                  const cls = pct >= 25 ? 'green' : pct >= 10 ? 'cyan' : 'dim';
                  mined = <span className={cls}>{pct.toFixed(1)}%</span>;
                }

                const error = String(v.error_message || '').trim();

                return (
                  <tr key={v.video_id}>
                    <td style={{ fontFamily: 'var(--font-mono)', fontSize: 11, whiteSpace: 'nowrap' }}>
                      <button className="link-btn" title="Open the MongoDB document"
                              onClick={() => setOpenVideo(v.video_id)}>
                        {v.video_id || ''}
                      </button>
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      {v.youtube_url ? (
                        <a
                          className="yt-link"
                          href={v.youtube_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          title={v.youtube_url}
                        >
                          ▶
                        </a>
                      ) : (
                        <span className="dim">—</span>
                      )}
                    </td>
                    <td className="cell-title" title={title}>
                      {title ? (title.length > 60 ? `${title.slice(0, 60)}…` : title) : <span className="dim">—</span>}
                    </td>
                    {shown.has('region') && <Cell value={v.region} />}
                    <td><span className={`badge ${status}`}>{status}</span></td>
                    {shown.has('source') && <Cell value={v.source} />}
                    {shown.has('source_channel') && <Cell value={v.source_channel} />}
                    {shown.has('license') && <Cell value={v.license} />}
                    <Cell numeric value={Number.isFinite(srcS) && srcS > 0 ? formatDurationShort(srcS) : '—'} />
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>
                      {Number.isFinite(extS) && extS > 0 ? formatDurationShort(extS) : '—'}
                    </td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>{mined}</td>
                    <Cell numeric value={v.total_segments || '—'} />
                    {shown.has('avg_asd_score') && (
                      <Cell numeric value={v.avg_asd_score ? parseFloat(v.avg_asd_score).toFixed(2) : '—'} />
                    )}
                    {shown.has('avg_syncnet_conf') && (
                      <Cell numeric value={v.avg_syncnet_conf ? parseFloat(v.avg_syncnet_conf).toFixed(2) : '—'} />
                    )}
                    {shown.has('gender') && <Cell value={v.gender} />}
                    {shown.has('age_group') && <Cell value={v.age_group} />}
                    {shown.has('environment') && <Cell value={v.environment} />}
                    {shown.has('background_noise') && <Cell value={v.background_noise} />}
                    {shown.has('num_speakers') && <Cell numeric value={v.num_speakers || '—'} />}
                    {shown.has('processed_date') && <Cell value={v.processed_date} />}
                    {shown.has('error_message') && (
                      <td>
                        {error ? (
                          <span className="badge failed" title={error}>
                            {error.slice(0, 28)}{error.length > 28 ? '…' : ''}
                          </span>
                        ) : (
                          <span className="dim">—</span>
                        )}
                      </td>
                    )}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
      <Pagination page={safePage} totalPages={totalPages} onPageChange={setPage} />
      {openVideo && <VideoDocumentModal videoId={openVideo} onClose={() => setOpenVideo(null)} />}
    </div>
  );
}

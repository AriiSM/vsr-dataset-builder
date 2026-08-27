import { useState, useMemo } from 'react';
import { formatDurationShort } from '../../utils/format.js';
import { useDebouncedValue } from '../../hooks/useDebouncedValue.js';
import { Pagination } from '../../components/Pagination.jsx';

const VIDEOS_PER_PAGE = 25;

/** The status filters available above the table. */
const STATUS_FILTERS = [
  { value: 'all',       label: 'All' },
  { value: 'completed', label: 'Done' },
  { value: 'validated', label: 'Validated' },
  { value: 'pending',   label: 'Pending' },
  { value: 'failed',    label: 'Failed' },
];

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
 * by ID/title, a status filter, and pagination. The table is wide (21 columns)
 * → horizontal scrolling inside the section.
 */
export function VideosTable({ videos, headerActions = null }) {
  const [statusFilter, setStatusFilter] = useState('all');
  const [searchInput, setSearchInput] = useState('');
  const [page, setPage] = useState(1);
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

  const totalPages = Math.max(1, Math.ceil(filtered.length / VIDEOS_PER_PAGE));
  const safePage = Math.min(page, totalPages);
  const pageRows = filtered.slice((safePage - 1) * VIDEOS_PER_PAGE, safePage * VIDEOS_PER_PAGE);

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

      <div className="cmd-table-wrap browse-table-scroll">
        <table className="videos-table">
          <thead>
            <tr>
              <th>Video ID</th>
              <th title="Open on YouTube">YT</th>
              <th>Title</th>
              <th>Region</th>
              <th>Status</th>
              <th title="Source type (Interview / TEDx / Lecture / ...)">Type</th>
              <th>Channel</th>
              <th>License</th>
              <th title="Original source-video duration">Source</th>
              <th title="Total duration of exported segments">Extracted</th>
              <th title="Extracted ÷ Source — fraction of raw video that became dataset material">Mined%</th>
              <th title="Segment count">Segs</th>
              <th title="Average Active Speaker Detection score">ASD</th>
              <th title="Average SyncNet audio-visual sync confidence">SyncNet</th>
              <th title="Speaker gender">Gen</th>
              <th title="Speaker age group">Age</th>
              <th title="Recording environment">Env</th>
              <th title="Background noise level">Noise</th>
              <th title="Number of distinct speakers">#Spk</th>
              <th>Processed</th>
              <th title="Pipeline error message, if processing failed">Error</th>
            </tr>
          </thead>
          <tbody>
            {pageRows.length === 0 ? (
              <tr>
                <td colSpan={21} style={{ textAlign: 'center', padding: 24, color: 'var(--text-muted)' }}>
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
                      {v.video_id || ''}
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
                    <Cell value={v.region} />
                    <td><span className={`badge ${status}`}>{status}</span></td>
                    <Cell value={v.source} />
                    <Cell value={v.source_channel} />
                    <Cell value={v.license} />
                    <Cell numeric value={Number.isFinite(srcS) && srcS > 0 ? formatDurationShort(srcS) : '—'} />
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>
                      {Number.isFinite(extS) && extS > 0 ? formatDurationShort(extS) : '—'}
                    </td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>{mined}</td>
                    <Cell numeric value={v.total_segments || '—'} />
                    <Cell numeric value={v.avg_asd_score ? parseFloat(v.avg_asd_score).toFixed(2) : '—'} />
                    <Cell numeric value={v.avg_syncnet_conf ? parseFloat(v.avg_syncnet_conf).toFixed(2) : '—'} />
                    <Cell value={v.gender} />
                    <Cell value={v.age_group} />
                    <Cell value={v.environment} />
                    <Cell value={v.background_noise} />
                    <Cell numeric value={v.num_speakers || '—'} />
                    <Cell value={v.processed_date} />
                    <td>
                      {error ? (
                        <span className="badge failed" title={error}>
                          {error.slice(0, 28)}{error.length > 28 ? '…' : ''}
                        </span>
                      ) : (
                        <span className="dim">—</span>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
      <Pagination page={safePage} totalPages={totalPages} onPageChange={setPage} />
    </div>
  );
}

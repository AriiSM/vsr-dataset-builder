import { useState, useMemo } from 'react';
import { formatDuration } from '../../utils/format.js';

// How many rows we actually render: the vocabulary can have tens of thousands
// of words, and a DOM with that many <tr> freezes the page. Sorting/searching
// run over the full set; only the first MAX_RENDERED_ROWS results are shown.
const MAX_RENDERED_ROWS = 500;

/** The vocabulary table's columns (all sortable). */
const COLUMNS = [
  { key: 'word',        label: 'Word',   title: undefined,                                              width: undefined },
  { key: 'len',         label: 'Len',    title: 'Letter count',                                         width: 48 },
  { key: 'samples',     label: '#Seg',   title: 'Distinct segments containing this word',               width: 60 },
  { key: 'occurrences', label: '#Total', title: 'Total occurrences across the corpus',                  width: 60 },
  { key: 'share',       label: '%',      title: 'Share of all spoken words (%)',                        width: 60 },
  { key: 'duration',    label: 'Dur',    title: 'Total spoken duration of this word across the corpus', width: 80 },
  { key: 'avg_dur',     label: 'Avg',    title: 'Average spoken duration per occurrence (ms)',          width: 64 },
];

/**
 * The vocabulary table: each word with its frequency, share, and spoken
 * duration. Search + sorting on any column.
 */
export function VocabTable({ words, totalUnique }) {
  const [filter, setFilter] = useState('');
  const [sortKey, setSortKey] = useState('samples');
  const [sortAsc, setSortAsc] = useState(false);

  // The derived columns (length, share, average duration) are computed
  // once per data set; after that we only filter/sort.
  const enriched = useMemo(() => {
    const totalOccurrences = words.reduce((sum, w) => sum + (w.occurrences || 0), 0) || 1;
    return words.map((w) => ({
      ...w,
      len: w.word.length,
      share: ((w.occurrences || 0) / totalOccurrences) * 100,
      avg_dur: (w.occurrences || 0) > 0 ? w.duration / w.occurrences : 0,
    }));
  }, [words]);

  const visible = useMemo(() => {
    const q = filter.toUpperCase();
    const data = q ? enriched.filter((w) => w.word.includes(q)) : [...enriched];
    data.sort((a, b) => {
      let va = a[sortKey];
      let vb = b[sortKey];
      if (typeof va === 'string') { va = va.toLowerCase(); vb = vb.toLowerCase(); }
      if (va < vb) return sortAsc ? -1 : 1;
      if (va > vb) return sortAsc ? 1 : -1;
      return 0;
    });
    return data;
  }, [enriched, filter, sortKey, sortAsc]);

  /** Header click: the same column flips the direction, another one selects it. */
  function handleSort(key) {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(key === 'word'); // text starts alphabetical, numerics descending
    }
  }

  return (
    <div className="browse-section browse-section-vocab">
      <div className="cmd-panel-hdr">
        <span className="cmd-panel-label">Vocabulary</span>
        <span className="cmd-panel-count">{`${(totalUnique || 0).toLocaleString()} unique`}</span>
        <div className="cmd-right-actions">
          <input
            type="text"
            placeholder="Filter words..."
            className="vocab-search-input"
            style={{ width: 150 }}
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
        </div>
      </div>
      <div className="cmd-table-wrap browse-table-scroll">
        <table className="vocab-table">
          <thead>
            <tr>
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  title={col.title}
                  style={{ cursor: 'pointer', ...(col.width ? { width: col.width } : {}) }}
                  onClick={() => handleSort(col.key)}
                >
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visible.slice(0, MAX_RENDERED_ROWS).map((w) => (
              <tr key={w.word}>
                <td>{w.word}</td>
                <td className="dim">{w.len}</td>
                <td>{(w.samples || 0).toLocaleString()}</td>
                <td>{(w.occurrences || 0).toLocaleString()}</td>
                <td className="dim">{w.share.toFixed(3)}%</td>
                <td>{formatDuration(w.duration)}</td>
                <td className="dim">{(w.avg_dur * 1000).toFixed(0)} ms</td>
              </tr>
            ))}
            {visible.length > MAX_RENDERED_ROWS && (
              <tr>
                <td colSpan={7} className="dim" style={{ textAlign: 'center', padding: 10 }}>
                  Showing first {MAX_RENDERED_ROWS} of {visible.length.toLocaleString()} words —
                  use the filter to narrow down.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

import { useState, useMemo, useEffect } from 'react';
import { api } from '../../api.js';
import { toast } from '../../components/toast.jsx';
import { formatDurationShort } from '../../utils/format.js';
import { Pagination } from '../../components/Pagination.jsx';
import { detectSpeakerCapabilities } from '../../utils/datasetCapabilities.js';

const SPEAKERS_PER_PAGE = 25;

/** The numeric columns — for these, the default sort is descending. */
const NUMERIC_KEYS = new Set(['num_videos', 'num_segments', 'total_duration_s', 'avg_asd', 'avg_wer', 'age_estimate', 'gender_confidence']);

/** The table's sortable columns (key + label). */
const COLUMNS = [
  { key: 'speaker_id',       label: 'Speaker ID' },
  { key: 'speaker_name',     label: 'Name' },
  { key: 'gender',           label: 'Gender' },
  { key: 'age_group',        label: 'Age' },
  { key: 'accent_region',    label: 'Accent' },
  { key: 'num_videos',       label: '#Vid' },
  { key: 'num_segments',     label: '#Seg' },
  { key: 'total_duration_s', label: 'Dur' },
  { key: 'avg_asd',          label: 'ASD' },
  { key: 'avg_wer',          label: 'WER' },
];

/** A speaker's sort value for a column (numeric or text). */
function sortValue(speaker, key) {
  const v = speaker[key];
  if (v === undefined || v === null || v === '') {
    return NUMERIC_KEYS.has(key) ? -Infinity : '';
  }
  if (NUMERIC_KEYS.has(key)) {
    const n = parseFloat(v);
    return Number.isFinite(n) ? n : -Infinity;
  }
  return String(v).toLowerCase();
}

/** A speaker's thumbnail, with a fallback if the image is missing on disk. */
function SpeakerThumb({ speakerId, seg = 0, cacheBust = '' }) {
  const [failed, setFailed] = useState(false);
  if (failed) return <span className="speaker-thumb-missing" title="no preview"></span>;
  return (
    <img
      className="speaker-thumb"
      src={api.speakerThumbnailUrl(speakerId, seg, cacheBust)}
      alt=""
      loading="lazy"
      onError={() => setFailed(true)}
    />
  );
}

/**
 * The speaker edit dialog: name, gender, age, accent + a gallery of
 * 4 thumbnails for visual confirmation (is it the same person in all of them?).
 */
function SpeakerModal({ speaker, onClose, onSaved }) {
  const [form, setForm] = useState({
    speaker_name: speaker.speaker_name || '',
    gender: speaker.gender || '',
    age_group: speaker.age_group || '',
    accent_region: speaker.accent_region || '',
  });
  // Cache-bust once per open, in case clustering was re-run.
  const [cacheBust] = useState(() => String(Date.now()));

  // Escape closes the dialog.
  useEffect(() => {
    const handleKey = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose]);

  const set = (field) => (e) => setForm({ ...form, [field]: e.target.value });

  async function handleSave() {
    try {
      const { ok, data } = await api.updateSpeaker(speaker.speaker_id, {
        ...form,
        speaker_name: form.speaker_name.trim(),
      });
      if (!ok) {
        toast.error(`Speaker save failed: ${data.error || 'unknown error'}`);
        return;
      }
      onSaved();
    } catch (err) {
      toast.error(`Speaker save error: ${err}`);
    }
  }

  return (
    <div
      className="speaker-modal-backdrop"
      style={{ display: 'flex' }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="speaker-modal"
        role="dialog"
        aria-modal="true"
        aria-label={`Edit speaker ${speaker.speaker_id}`}
      >
        <div className="speaker-modal-hdr">
          <span className="speaker-modal-title">
            Edit speaker <span className="mono">{speaker.speaker_id}</span>
          </span>
          <button className="btn-micro" onClick={onClose}>✕</button>
        </div>
        <div className="speaker-modal-thumbs">
          {[0, 1, 2, 3].map((seg) => (
            <SpeakerThumb key={seg} speakerId={speaker.speaker_id} seg={seg} cacheBust={cacheBust} />
          ))}
        </div>
        <div className="speaker-modal-body">
          <label>
            Name
            <input type="text" placeholder="Andreea Esca" value={form.speaker_name} onChange={set('speaker_name')} />
          </label>
          <label>
            Gender
            <select value={form.gender} onChange={set('gender')}>
              <option value="">—</option>
              <option value="M">M</option>
              <option value="F">F</option>
              <option value="mixed">mixed</option>
              <option value="unknown">unknown</option>
            </select>
          </label>
          <label>
            Age group
            <select value={form.age_group} onChange={set('age_group')}>
              <option value="">—</option>
              <option value="18-30">18-30</option>
              <option value="31-50">31-50</option>
              <option value="51+">51+</option>
              <option value="mixed">mixed</option>
              <option value="unknown">unknown</option>
            </select>
          </label>
          <label>
            Accent region
            <select value={form.accent_region} onChange={set('accent_region')}>
              <option value="">—</option>
              <option value="RO">RO</option>
              <option value="MD">MD</option>
              <option value="DIASPORA">DIASPORA</option>
              <option value="UNKNOWN">UNKNOWN</option>
            </select>
          </label>
        </div>
        <div className="speaker-modal-actions">
          <button className="btn-micro" onClick={onClose}>Cancel</button>
          <button className="btn-micro primary" onClick={handleSave}>Save</button>
        </div>
      </div>
    </div>
  );
}

/**
 * The speakers table: sorting on any column, pagination, and editing
 * via a modal (click on the row or on the Edit button).
 */
export function SpeakersTable({ speakers, onSpeakersChanged }) {
  const [sort, setSort] = useState({ key: 'num_segments', asc: false }); // most active first
  // v3 columns (numeric age, gender confidence, identity match) appear only
  // when the registry actually carries them.
  const caps = useMemo(() => detectSpeakerCapabilities(speakers), [speakers]);
  const [page, setPage] = useState(1);
  const [editingSpeaker, setEditingSpeaker] = useState(null);

  const sorted = useMemo(() => {
    const dir = sort.asc ? 1 : -1;
    return [...speakers].sort((a, b) => {
      const av = sortValue(a, sort.key);
      const bv = sortValue(b, sort.key);
      if (av < bv) return -1 * dir;
      if (av > bv) return 1 * dir;
      return 0;
    });
  }, [speakers, sort]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / SPEAKERS_PER_PAGE));
  const safePage = Math.min(page, totalPages);
  const pageRows = sorted.slice((safePage - 1) * SPEAKERS_PER_PAGE, safePage * SPEAKERS_PER_PAGE);

  /** Header click: switches the column or flips the direction. */
  function handleHeaderClick(key) {
    setPage(1);
    setSort((prev) =>
      prev.key === key
        ? { key, asc: !prev.asc }
        // Default: descending for numerics, alphabetical for text.
        : { key, asc: !NUMERIC_KEYS.has(key) }
    );
  }

  return (
    <div className="browse-section browse-section-speakers">
      <div className="cmd-panel-hdr">
        <span className="cmd-panel-label">Speakers</span>
        <span className="cmd-panel-count">
          {sorted.length ? `${sorted.length} speaker(s) · page ${safePage}/${totalPages}` : ''}
        </span>
      </div>
      <div className="cmd-table-wrap">
        <table className="speakers-table">
          <thead>
            <tr>
              <th></th>
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  className={`sortable ${sort.key === col.key ? (sort.asc ? 'sort-asc' : 'sort-desc') : ''}`}
                  onClick={() => handleHeaderClick(col.key)}
                >
                  {col.label}
                </th>
              ))}
              {caps.hasAgeEstimate && (
                <th
                  className={`sortable ${sort.key === 'age_estimate' ? (sort.asc ? 'sort-asc' : 'sort-desc') : ''}`}
                  title="Numeric age estimate ± spread; a large spread means the samples disagree"
                  onClick={() => handleHeaderClick('age_estimate')}
                >
                  Age est.
                </th>
              )}
              {caps.hasGenderConfidence && (
                <th
                  className={`sortable ${sort.key === 'gender_confidence' ? (sort.asc ? 'sort-asc' : 'sort-desc') : ''}`}
                  title="Share of samples that voted for the majority gender"
                  onClick={() => handleHeaderClick('gender_confidence')}
                >
                  G-conf
                </th>
              )}
              <th></th>
            </tr>
          </thead>
          <tbody>
            {pageRows.length === 0 ? (
              <tr>
                <td colSpan={12 + (caps.hasAgeEstimate ? 1 : 0) + (caps.hasGenderConfidence ? 1 : 0)} style={{ textAlign: 'center', padding: 18, color: 'var(--text-muted)' }}>
                  No speakers in registry yet. They are auto-created when a video is processed.
                </td>
              </tr>
            ) : (
              pageRows.map((sp) => {
                const dur = parseFloat(sp.total_duration_s || 0) || 0;
                const wer = sp.avg_wer === '' || sp.avg_wer == null
                  ? '—'
                  : `${(parseFloat(sp.avg_wer) * 100).toFixed(1)}%`;
                return (
                  <tr key={sp.speaker_id} onClick={() => setEditingSpeaker(sp)}>
                    <td><SpeakerThumb speakerId={sp.speaker_id} /></td>
                    <td className="mono" style={{ fontSize: 11 }}>
                      {sp.speaker_id}
                      {String(sp.identity_match || '').trim() === 'auto' && (
                        <span
                          className="identity-badge"
                          title="Matched across videos automatically (ArcFace) — verify in Review"
                        >
                          auto
                        </span>
                      )}
                    </td>
                    <td>{sp.speaker_name || ''}</td>
                    <td>{sp.gender || ''}</td>
                    <td>{sp.age_group || ''}</td>
                    <td>{sp.accent_region || ''}</td>
                    <td>{sp.num_videos || 0}</td>
                    <td>{sp.num_segments || 0}</td>
                    <td>{formatDurationShort(dur)}</td>
                    <td>{sp.avg_asd || '—'}</td>
                    <td>{wer}</td>
                    {caps.hasAgeEstimate && (() => {
                      const age = parseFloat(sp.age_estimate);
                      const std = parseFloat(sp.age_std);
                      if (!Number.isFinite(age)) return <td className="dim">—</td>;
                      // A wide spread means the per-sample estimates disagree
                      // — flag it for manual verification.
                      const uncertain = Number.isFinite(std) && std > 8;
                      return (
                        <td className={uncertain ? 'value-warn' : ''}>
                          {Math.round(age)}{Number.isFinite(std) ? ` ±${Math.round(std)}` : ''}
                        </td>
                      );
                    })()}
                    {caps.hasGenderConfidence && (() => {
                      const conf = parseFloat(sp.gender_confidence);
                      if (!Number.isFinite(conf)) return <td className="dim">—</td>;
                      const pct = conf <= 1 ? conf * 100 : conf;
                      return <td>{pct.toFixed(0)}%</td>;
                    })()}
                    <td>
                      <button
                        className="btn-micro speakers-edit-btn"
                        onClick={(e) => { e.stopPropagation(); setEditingSpeaker(sp); }}
                      >
                        Edit
                      </button>
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
      <Pagination page={safePage} totalPages={totalPages} onPageChange={setPage} />

      {editingSpeaker && (
        <SpeakerModal
          speaker={editingSpeaker}
          onClose={() => setEditingSpeaker(null)}
          onSaved={() => {
            setEditingSpeaker(null);
            onSpeakersChanged();  // reload the list so the aggregates are fresh
          }}
        />
      )}
    </div>
  );
}

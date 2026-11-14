/**
 * The two summary blocks of the Overview page:
 *   GoalHero       — the distance to the 200 h goal, at each quality level
 *   ReviewProgress — the manual-review counters
 * Display only: both receive the already-loaded /api/stats payload.
 */
import { formatDurationShort } from '../../utils/format.js';
import { TARGET_REGION, TARGET_HOURS } from '../../config.js';
import { MetricRows } from '../../components/MetricRows.jsx';

const HOUR = 3600;

/** Extracted seconds for the target region (segment breakdown first). */
function targetRegionSeconds(stats) {
  const segRegion = (stats.segments || {}).by_region?.[TARGET_REGION];
  if (segRegion?.duration_s != null) return segRegion.duration_s;
  return (stats.videos || {}).speech_by_region_s?.[TARGET_REGION] || 0;
}

const hours = (secs) => `${(secs / HOUR).toFixed(1)} h`;

/**
 * One track, 0 → 200 h, with the quality levels drawn on top of each other:
 * every level is a subset of the one before it, so the bars nest and the
 * gap from each bar's end to the goal line is what is still missing.
 */
export function GoalHero({ stats }) {
  const s = stats.segments || {};
  const v = stats.videos || {};
  const funnel = s.training_funnel || [];
  const step = (key) => funnel.find((f) => f.key === key);

  const levels = [
    {
      key: 'extracted', cls: 'lvl-extracted',
      name: 'Extracted',
      desc: `${TARGET_REGION} speech cut into clips`,
      secs: targetRegionSeconds(stats),
      clips: s.total,
    },
    {
      key: 'candidates', cls: 'lvl-candidates',
      name: 'Pass the filters',
      desc: 'speaker visible, usable text, 1–16 s',
      secs: step('length')?.duration_s,
      clips: step('length')?.count,
    },
    {
      key: 'agreed', cls: 'lvl-agreed',
      name: 'Transcripts agree',
      desc: 'Whisper variants give the same text',
      secs: step('agreed')?.duration_s,
      clips: step('agreed')?.count,
    },
    {
      key: 'approved', cls: 'lvl-approved',
      name: 'Approved by hand',
      desc: 'checked in Review',
      secs: (s.approved || {}).total_duration_s || 0,
      clips: (s.approved || {}).count || 0,
    },
  ].filter((l) => l.secs != null);

  const goalS = TARGET_HOURS * HOUR;
  const scaleS = Math.max(goalS, ...levels.map((l) => l.secs));
  const pct = (secs) => `${Math.min(100, (secs / scaleS) * 100).toFixed(2)}%`;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * TARGET_HOURS);
  const extracted = levels[0]?.secs || 0;
  const sourceS = v.total_source_duration_s ?? (v.total_source_duration_h || 0) * HOUR;

  return (
    <section className="goal-hero" aria-label={`Progress toward ${TARGET_HOURS} hours`}>
      <div className="goal-hero-head">
        <div className="goal-hero-figure">
          <span className="goal-hero-num">{(extracted / HOUR).toFixed(1)}</span>
          <span className="goal-hero-of">of {TARGET_HOURS} hours extracted</span>
        </div>
        <p className="goal-hero-context">
          From {formatDurationShort(sourceS)} of video in {(v.total || 0).toLocaleString()} files.
          The bars below are nested: each level is part of the one above it.
        </p>
      </div>

      <div className="goal-track-wrap">
        <div className="goal-track">
          {levels.map((l) => (
            <div key={l.key} className={`goal-bar ${l.cls}`} style={{ width: pct(l.secs) }} />
          ))}
          <div className="goal-line" style={{ left: pct(goalS) }} />
        </div>
        <div className="goal-ticks" aria-hidden="true">
          {ticks.map((t) => (
            <span key={t} style={{ left: pct(t * HOUR) }}>{t === TARGET_HOURS ? `${t} h goal` : `${t} h`}</span>
          ))}
        </div>
      </div>

      <ul className="goal-legend">
        {levels.map((l) => (
          <li key={l.key} className={l.cls}>
            <span className="goal-swatch" aria-hidden="true" />
            <span className="goal-legend-text">
              <span className="goal-legend-name">{l.name}</span>
              <span className="goal-legend-desc">{l.desc}</span>
            </span>
            <span className="goal-legend-val">
              <strong>{l.secs > 0 ? hours(l.secs) : '0 h'}</strong>
              <span>
                {l.clips != null ? `${l.clips.toLocaleString()} clips` : ''}
                {l.secs > 0 ? ` · ${((l.secs / goalS) * 100).toFixed(0)}% of goal` : ''}
              </span>
            </span>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** Manual review counters: one row of plain figures. */
export function ReviewProgress({ stats }) {
  const s = stats.segments || {};
  const total = s.total || 0;
  const approved = (s.approved || {}).count || 0;
  const rejected = s.rejected_count || 0;
  const edited = s.edited_count || 0;
  const reviewed = approved + rejected;
  const pending = Math.max(0, total - reviewed);
  const pct = (n) => (total ? `${((n / total) * 100).toFixed(1)}%` : '—');
  // WER is measured against the curator's text, so it means nothing before
  // the first review.
  const werReady = s.avg_wer != null && approved + edited + rejected > 0;

  const items = [
    { label: 'Reviewed', value: reviewed.toLocaleString(), sub: `${pct(reviewed)} of ${total.toLocaleString()} clips` },
    { label: 'Waiting', value: pending.toLocaleString(), sub: 'not yet opened in Review' },
    { label: 'Approved', value: approved.toLocaleString(), sub: `${hours((s.approved || {}).total_duration_s || 0)} of speech` },
    { label: 'Text edited', value: edited.toLocaleString(), sub: 'transcript corrected by hand' },
    { label: 'Rejected', value: rejected.toLocaleString(), sub: 'media removed from disk' },
    {
      label: 'Average WER',
      value: werReady ? `${(s.avg_wer * 100).toFixed(1)}%` : '—',
      sub: werReady ? 'Whisper vs. corrected text' : 'available after the first review',
    },
  ];

  return (
    <div className="review-row span-12">
      <div className="review-meter" aria-hidden="true">
        <span style={{ width: total ? `${(reviewed / total) * 100}%` : 0 }} />
      </div>
      <dl className="review-figures">
        {items.map((it) => (
          <div key={it.label}>
            <dt>{it.label}</dt>
            <dd>
              <strong>{it.value}</strong>
              <span>{it.sub}</span>
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

/** How much usable speech the processed video yielded. */
export function YieldCard({ stats }) {
  const s = stats.segments || {};
  const v = stats.videos || {};
  const sourceS = v.total_source_duration_s ?? (v.total_source_duration_h || 0) * HOUR;
  const extractedS = s.total_duration_s ?? (s.total_duration_h || 0) * HOUR;
  const done = (v.by_status || {}).completed || 0;
  return (
    <MetricRows
      rows={[
        { label: 'Video processed', value: formatDurationShort(sourceS) },
        { label: 'Speech kept as clips', value: formatDurationShort(extractedS) },
        { label: 'Yield', value: sourceS ? `${((extractedS / sourceS) * 100).toFixed(1)}% of video` : '—' },
        { label: 'Average clip', value: `${(s.avg_duration_s || 0).toFixed(1)} s` },
        { label: 'Speech per finished video', value: done ? formatDurationShort(extractedS / done) : '—' },
      ]}
    />
  );
}

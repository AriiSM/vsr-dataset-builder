/**
 * The cards in the Stats tab's "Insights" grid.
 *
 * Each card is a small component that receives already-loaded data
 * (stats / distributions / speakers) and only displays it — no component
 * in here makes HTTP calls.
 */
import { formatDuration, formatDurationShort } from '../../utils/format.js';
import { TARGET_REGION, TARGET_HOURS } from '../../config.js';
import { TIER_META } from '../../utils/datasetCapabilities.js';
import { MetricRows, EmptyNote, CardSubheading } from '../../components/MetricRows.jsx';
import { RankList } from '../../components/RankList.jsx';

/**
 * Extracted seconds for the target region (MD).
 * Prefers the segment-level breakdown (recomputed from segments_index, so it
 * survives rejects) and falls back to the videos_master aggregate.
 */
function targetRegionSeconds(stats) {
  const segRegion = (stats.segments || {}).by_region?.[TARGET_REGION];
  if (segRegion?.duration_s != null) return segRegion.duration_s;
  return (stats.videos || {}).speech_by_region_s?.[TARGET_REGION] || 0;
}

/* ────────────────────────────────────────────────────────────────────────
   Row 1 — Pipeline & status
   ──────────────────────────────────────────────────────────────────────── */

/** The pipeline funnel: Source → Extracted → Approved, with conversion rates. */
export function PipelineFunnelCard({ stats }) {
  const segments = stats.segments || {};
  const videos = stats.videos || {};
  const approved = segments.approved || {};

  const sourceDurS = videos.total_source_duration_s ?? (videos.total_source_duration_h || 0) * 3600;
  const extractedDurS = segments.total_duration_s ?? (segments.total_duration_h || 0) * 3600;
  const approvedDurS = approved.total_duration_s || 0;
  const totalClips = segments.total || 0;

  // Bar widths are relative to the source duration (the largest) so the
  // funnel visually narrows as the data gets filtered.
  const referenceDur = Math.max(sourceDurS, 1);
  const widthPct = (dur) => `${((dur / referenceDur) * 100).toFixed(2)}%`;

  const minedPct = sourceDurS > 0 ? (extractedDurS / sourceDurS) * 100 : 0;
  const approvedPct = extractedDurS > 0 ? (approvedDurS / extractedDurS) * 100 : 0;

  const steps = [
    {
      cls: 'source', name: 'Source', value: formatDurationShort(sourceDurS),
      valueCls: '', barCls: 'muted', barWidth: widthPct(sourceDurS),
      sub: `${(videos.total || 0).toLocaleString()} files`,
    },
    {
      cls: 'extracted', name: 'Extracted', value: formatDurationShort(extractedDurS),
      valueCls: 'cyan', barCls: 'cyan', barWidth: widthPct(extractedDurS),
      sub: `${totalClips.toLocaleString()} clips · avg ${(segments.avg_duration_s || 0).toFixed(1)}s`,
    },
    {
      cls: 'approved', name: 'Approved', value: formatDurationShort(approvedDurS),
      valueCls: 'green', barCls: 'green', barWidth: widthPct(approvedDurS),
      sub: `${(approved.count || 0).toLocaleString()} clips`,
    },
  ];
  const arrows = [`mined ${minedPct.toFixed(1)}%`, `approved ${approvedPct.toFixed(1)}%`];

  return (
    <div className="funnel">
      {steps.map((step, i) => (
        <div key={step.cls}>
          {i > 0 && <div className="funnel-arrow">↓ {arrows[i - 1]}</div>}
          <div className={`funnel-step ${step.cls}`}>
            <div className="funnel-hdr">
              <span className="funnel-name">{step.name}</span>
              <span className={`funnel-value ${step.valueCls}`}>{step.value}</span>
            </div>
            <div className="funnel-bar">
              <div className={`funnel-bar-fill ${step.barCls}`} style={{ width: step.barWidth }} />
            </div>
            <div className="funnel-sub">{step.sub}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

/** Videos by status: stacked bar + one row per status. */
export function VideoStatusCard({ stats }) {
  const videos = stats.videos || {};
  const byStatus = videos.by_status || {};
  const total = videos.total || 0;

  const ORDER = [
    { key: 'completed',  label: 'Completed',  cls: 'green' },
    { key: 'validated',  label: 'Validated',  cls: 'cyan' },
    { key: 'pending',    label: 'Pending',    cls: 'orange' },
    { key: 'processing', label: 'Processing', cls: 'orange' },
    { key: 'failed',     label: 'Failed',     cls: 'red' },
    { key: 'skipped',    label: 'Skipped',    cls: 'muted' },
  ];
  const present = ORDER.filter((o) => byStatus[o.key]);
  const pct = (n) => (total > 0 ? (n / total) * 100 : 0);

  return (
    <>
      <div className="progress-bar">
        {present.map((o) => (
          <div
            key={o.key}
            className={`progress-fill ${o.cls}`}
            style={{ width: `${pct(byStatus[o.key]).toFixed(2)}%` }}
            title={`${o.label} ${byStatus[o.key]}`}
          />
        ))}
      </div>
      <div className="status-rows">
        {present.length === 0 ? (
          <div className="dim" style={{ padding: '8px 0' }}>No videos yet</div>
        ) : (
          present.map((o) => (
            <div className="status-row" key={o.key}>
              <span className="status-label"><i className={`dot ${o.cls}`}></i>{o.label}</span>
              <div className="status-bar">
                <div className={`progress-fill ${o.cls}`} style={{ width: `${pct(byStatus[o.key]).toFixed(2)}%` }} />
              </div>
              <span className="status-count mono">
                {byStatus[o.key].toLocaleString()} <span className="dim">/ {total}</span>
              </span>
            </div>
          ))
        )}
      </div>
    </>
  );
}

/** Per-segment quality averages (words, ASD, Whisper, characters). */
export function QualityMetricsCard({ stats }) {
  const s = stats.segments || {};
  if (!s.total) return <EmptyNote>No segments yet</EmptyNote>;
  return (
    <MetricRows
      rows={[
        { label: 'Avg words per segment',  value: (s.avg_words || 0).toFixed(1) },
        { label: 'Avg ASD score',          value: (s.avg_asd || 0).toFixed(2) },
        { label: 'Avg SyncNet confidence', value: (s.avg_syncnet || 0).toFixed(2) },
        { label: 'Avg Whisper confidence', value: (s.avg_whisper_conf || 0).toFixed(2) },
        { label: 'Total characters',       value: (s.total_chars || 0).toLocaleString() },
      ]}
    />
  );
}

/** Data health: segments without speaker_id / WER, Conf 1 annotations. */
export function HealthCard({ distributions }) {
  const health = distributions?.health || {};
  if (!health.n_segments) return <EmptyNote>No data</EmptyNote>;

  const pctClass = (pct) => (pct > 50 ? 'bad' : pct > 20 ? 'warn' : '');
  const rows = [
    { label: 'Segments without speaker_id', count: health.missing_speaker_id.count, pct: health.missing_speaker_id.pct, suffix: '' },
    { label: 'Segments without WER', count: health.missing_wer.count, pct: health.missing_wer.pct, suffix: '' },
    { label: 'Conf 1 (low) annotations', count: health.conf_1_low.count, pct: health.conf_1_low.pct_of_scanned, suffix: ` of ${health.conf_1_low.annotations_scanned} scanned` },
  ];
  return (
    <>
      {rows.map((r) => (
        <div className="health-row" key={r.label}>
          <span>{r.label}</span>
          <span className="mono">
            {r.count.toLocaleString()}
            <span className={`health-pct ${pctClass(r.pct)}`}>{r.pct.toFixed(1)}%{r.suffix}</span>
          </span>
        </div>
      ))}
    </>
  );
}

/* ────────────────────────────────────────────────────────────────────────
   Row 2 — Corpus composition
   ──────────────────────────────────────────────────────────────────────── */

/** The clip duration profile: quantiles + the distribution across length bands. */
export function DurationProfileCard({ stats }) {
  const s = stats.segments || {};
  const quantiles = s.duration_stats || {};
  const bands = s.duration_bands || [];
  if (!s.total || !bands.length) return <EmptyNote>No segments yet</EmptyNote>;

  const fmtSec = (v) => (v != null ? `${v.toFixed(2)}s` : '—');
  const maxCount = Math.max(1, ...bands.map((b) => b.count));

  // The 3–6s band is the training "sweet spot" (green); the tails (<1s, >15s)
  // signal a poorly configured VAD (orange).
  const bandColor = (label) => {
    if (label === '3–6s') return 'var(--green)';
    if (label === '<1s' || label === '>15s') return 'var(--orange)';
    return undefined;
  };

  return (
    <>
      <MetricRows
        rows={[
          { label: 'Median',    value: fmtSec(quantiles.median) },
          { label: 'Mean',      value: fmtSec(quantiles.mean) },
          { label: 'p95',       value: fmtSec(quantiles.p95) },
          { label: 'Min … Max', value: quantiles.min != null ? `${fmtSec(quantiles.min)} … ${fmtSec(quantiles.max)}` : '—' },
          { label: 'Total',     value: `${s.total.toLocaleString()} clips · ${formatDurationShort(s.total_duration_s || 0)}` },
        ]}
      />
      <CardSubheading>By length band</CardSubheading>
      <RankList
        rows={bands.map((b) => ({
          key: b.label,
          label: b.label,
          value: `${b.count.toLocaleString()} · ${b.pct.toFixed(1)}%`,
          barPct: (b.count / maxCount) * 100,
          barColor: bandColor(b.label),
        }))}
      />
    </>
  );
}

/** Two stacked lists: videos by source type and by license. */
export function BySourceCard({ stats }) {
  const videos = stats.videos || {};
  const sections = [
    { title: 'Source type', data: videos.by_source || {} },
    { title: 'License', data: videos.by_license || {} },
  ].filter((sec) => Object.keys(sec.data).length > 0);

  if (!sections.length) return <EmptyNote>No breakdown</EmptyNote>;

  return (
    <>
      {sections.map((sec) => {
        const entries = Object.entries(sec.data);
        const max = Math.max(1, ...entries.map(([, v]) => v));
        return (
          <div key={sec.title}>
            <CardSubheading>{sec.title}</CardSubheading>
            <RankList
              rows={entries.map(([name, count]) => ({
                key: name,
                label: name,
                value: String(count),
                barPct: (count / max) * 100,
              }))}
            />
          </div>
        );
      })}
    </>
  );
}

/** Top 8 YouTube channels by video count. */
export function TopChannelsCard({ stats }) {
  const entries = Object.entries((stats.videos || {}).top_channels || {}).slice(0, 8);
  if (!entries.length) return <EmptyNote>No channel data yet</EmptyNote>;
  const max = Math.max(1, ...entries.map(([, v]) => v));
  return (
    <RankList
      rows={entries.map(([channel, count]) => ({
        key: channel || '—',
        label: channel || '—',
        value: `${count} vid${count === 1 ? '' : 's'}`,
        barPct: (count / max) * 100,
      }))}
    />
  );
}

/** Speaker demographics: by gender and by age group (count + duration). */
export function DemographicsCard({ speakers }) {
  if (!speakers.length) return <EmptyNote>No speakers yet</EmptyNote>;

  const GENDER_ORDER = ['M', 'F', 'mixed', 'unknown'];
  const AGE_ORDER = ['18-30', '31-50', '51+', 'mixed', 'unknown'];
  const LABELS = {
    M: 'Male', F: 'Female', mixed: 'Mixed', unknown: '—',
    '18-30': '18–30', '31-50': '31–50', '51+': '51+',
  };
  const BAR_CLASSES = {
    M: 'male', F: 'female', mixed: '', unknown: 'unknown',
    '18-30': 'young', '31-50': 'adult', '51+': 'senior',
  };

  // We aggregate by gender and by age at the same time; speaking duration
  // matters more than speaker count for the balance of a VSR dataset.
  const byGender = {};
  const byAge = {};
  let totalDur = 0;
  for (const sp of speakers) {
    const gender = String(sp.gender || '').trim() || 'unknown';
    const age = String(sp.age_group || '').trim() || 'unknown';
    const dur = parseFloat(sp.total_duration_s || 0) || 0;
    (byGender[gender] ??= { count: 0, dur: 0 });
    byGender[gender].count++;
    byGender[gender].dur += dur;
    (byAge[age] ??= { count: 0, dur: 0 });
    byAge[age].count++;
    byAge[age].dur += dur;
    totalDur += dur;
  }

  const block = (title, data, order) => {
    // Categories in canonical order, then anything else that showed up in the data.
    const keys = order.filter((k) => data[k]);
    for (const k of Object.keys(data)) if (!keys.includes(k)) keys.push(k);
    const maxDur = Math.max(1, ...keys.map((k) => data[k].dur));

    return (
      <div className="demo-block" key={title}>
        <div className="demo-title">
          <span>{title}</span>
          <span className="mono">{speakers.length} speakers</span>
        </div>
        {keys.map((k) => {
          const d = data[k];
          const sharePct = totalDur > 0 ? (d.dur / totalDur) * 100 : 0;
          return (
            <div className="demo-row" key={k}>
              <span className="demo-label">{LABELS[k] || k}</span>
              <div className="demo-bar-wrap">
                <div
                  className={`demo-bar-fill ${BAR_CLASSES[k] || ''}`}
                  style={{ width: `${((d.dur / maxDur) * 100).toFixed(1)}%` }}
                />
              </div>
              <span className="demo-counts">
                {d.count} · {formatDurationShort(d.dur)} ({sharePct.toFixed(0)}%)
              </span>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <>
      {block('By gender', byGender, GENDER_ORDER)}
      {block('By age group', byAge, AGE_ORDER)}
    </>
  );
}

/* ────────────────────────────────────────────────────────────────────────
   Row 3 — Per-clip quality / distributions
   ──────────────────────────────────────────────────────────────────────── */

/** Generic bucket-based histogram (used for WER and durations). */
export function HistogramCard({ buckets, totalLabel, axisLeft, axisRight, barClass }) {
  if (!buckets.length) return <EmptyNote>No data</EmptyNote>;
  const max = Math.max(1, ...buckets.map((b) => b.count));
  const total = buckets.reduce((sum, b) => sum + b.count, 0);
  return (
    <div className="dist-block">
      <div className="dist-title">
        <span>{totalLabel}</span>
        <span className="mono">{total}</span>
      </div>
      <div className="dist-bars">
        {buckets.map((b, i) => {
          const tip = b.lo_pct != null
            ? `${b.lo_pct}–${b.hi_pct}%: ${b.count}`
            : `${b.lo_s}–${b.hi_s}s: ${b.count}`;
          return (
            <div
              key={i}
              className={`dist-bar ${barClass}`}
              style={{ height: `${((b.count / max) * 100).toFixed(1)}%` }}
              title={tip}
            />
          );
        })}
      </div>
      <div className="dist-axis">
        <span>{axisLeft}</span>
        <span>{axisRight}</span>
      </div>
    </div>
  );
}

/** The distribution of confidence levels (Conf 1/2/3) from the annotations. */
export function ConfDistCard({ stats }) {
  // The backend serializes the keys as strings ("1"/"2"/"3"/"unknown").
  const d = (stats.segments || {}).by_conf || {};
  const total = (d['1'] || 0) + (d['2'] || 0) + (d['3'] || 0) + (d.unknown || 0);
  if (!total) return <EmptyNote>No annotation scan</EmptyNote>;

  const levels = [
    { label: 'High (3)',   cls: 'green',  count: d['3'] || 0 },
    { label: 'Medium (2)', cls: 'cyan',   count: d['2'] || 0 },
    { label: 'Low (1)',    cls: 'orange', count: d['1'] || 0 },
    { label: 'Unknown',    cls: 'muted',  count: d.unknown || 0 },
  ];
  return (
    <>
      <RankList
        rows={levels.map((l) => {
          const pct = (l.count / total) * 100;
          return {
            key: l.label,
            label: <><i className={`dot ${l.cls}`}></i>{l.label}</>,
            value: `${l.count.toLocaleString()} · ${pct.toFixed(1)}%`,
            barPct: pct,
            barColor: `var(--${l.cls === 'muted' ? 'text-muted' : l.cls})`,
          };
        })}
      />
      <div className="progress-sub" style={{ marginTop: 6 }}>Scanned up to 3,000 annotations.</div>
    </>
  );
}

/** Vocabulary coverage: total / unique / rare words / speaking pace. */
export function VocabSummaryCard({ stats }) {
  const s = stats.segments || {};
  const unique = s.unique_words || 0;
  const rareOnce = s.rare_words_1 || 0;
  const rareTwice = s.rare_words_2 || 0;
  const rarePct = (n) => (unique ? ((n / unique) * 100).toFixed(1) : 0);
  return (
    <MetricRows
      rows={[
        { label: 'Total words spoken',  value: (s.total_words || 0).toLocaleString() },
        { label: 'Unique vocabulary',   value: unique.toLocaleString() },
        { label: 'Words said once',     value: `${rareOnce.toLocaleString()} (${rarePct(rareOnce)}%)` },
        { label: 'Words said twice',    value: `${rareTwice.toLocaleString()} (${rarePct(rareTwice)}%)` },
        { label: 'Speaking pace',       value: `${s.words_per_minute || 0} words/min` },
        { label: 'Avg words / segment', value: (s.avg_words || 0).toFixed(1) },
      ]}
    />
  );
}

/* ────────────────────────────────────────────────────────────────────────
   Row 4 — Rankings & summaries
   ──────────────────────────────────────────────────────────────────────── */

/** Top videos, either by extracted duration or by "mining" ratio. */
export function TopVideosCard({ rows, getValue }) {
  if (!rows || !rows.length) return <EmptyNote>No data yet</EmptyNote>;
  const values = rows.map(getValue);
  const max = Math.max(1, ...values.map((v) => v.ref));
  return (
    <RankList
      rows={rows.map((r, i) => ({
        key: r.video_id,
        label: (
          <>
            <span className="rank-id">{r.video_id}</span>
            {r.title ? ` — ${r.title.slice(0, 40)}` : ''}
          </>
        ),
        value: values[i].label,
        barPct: (values[i].ref / max) * 100,
      }))}
    />
  );
}

/** Word and speech statistics (per-segment distribution). */
export function WordStatsCard({ stats }) {
  const s = stats.segments || {};
  const ws = s.words_stats || {};
  return (
    <MetricRows
      rows={[
        { label: 'Total words spoken',       value: (s.total_words || 0).toLocaleString() },
        { label: 'Words / segment (avg)',    value: (s.avg_words || 0).toFixed(1) },
        { label: 'Words / segment (median)', value: ws.median != null ? String(ws.median) : '—' },
        { label: 'Words / segment (min … max)', value: ws.min != null ? `${ws.min} … ${ws.max}` : '—' },
        { label: 'Speaking pace',            value: `${s.words_per_minute || 0} words/min` },
        { label: 'Total characters',         value: (s.total_chars || 0).toLocaleString() },
      ]}
    />
  );
}

/** Per-video duration ranges (min/mean/max, source and extracted). */
export function DurationRangesCard({ stats }) {
  const videos = stats.videos || {};
  const src = videos.source_duration || {};
  const ext = videos.extracted_duration || {};
  const cell = (v) => (v != null ? formatDurationShort(v) : '—');
  return (
    <MetricRows
      rows={[
        { label: 'Source · min',     value: cell(src.min) },
        { label: 'Source · mean',    value: cell(src.mean) },
        { label: 'Source · max',     value: cell(src.max) },
        { label: 'Extracted · min',  value: cell(ext.min) },
        { label: 'Extracted · mean', value: cell(ext.mean) },
        { label: 'Extracted · max',  value: cell(ext.max) },
      ]}
    />
  );
}

/**
 * v3 quality tiers (A/B/C) — count, duration and share per tier, as computed
 * by the quality_indexer service. Empty until the first v3 run writes the
 * quality_tier column.
 */
export function QualityTiersCard({ stats }) {
  const tiers = (stats.segments || {}).tiers;
  if (!tiers) {
    return <EmptyNote>No tier data yet — appears after the first pipeline v3 run</EmptyNote>;
  }
  const totalCount = ['A', 'B', 'C'].reduce((sum, t) => sum + (tiers[t]?.count || 0), 0);
  if (!totalCount) return <EmptyNote>No tiered segments yet</EmptyNote>;
  const maxCount = Math.max(1, ...['A', 'B', 'C'].map((t) => tiers[t]?.count || 0));

  return (
    <>
      <RankList
        rows={['A', 'B', 'C'].map((tierName) => {
          const tier = tiers[tierName] || { count: 0, duration_s: 0 };
          const sharePct = (tier.count / totalCount) * 100;
          return {
            key: tierName,
            label: (
              <>
                <i className={`dot ${TIER_META[tierName].cls}`}></i>
                {TIER_META[tierName].label}
              </>
            ),
            value: `${tier.count.toLocaleString()} · ${formatDurationShort(tier.duration_s)} · ${sharePct.toFixed(1)}%`,
            barPct: (tier.count / maxCount) * 100,
            barColor: TIER_META[tierName].color,
          };
        })}
      />
      <div className="progress-sub" style={{ marginTop: 6 }}>
        Tier A/B = usable for training; tier C = flagged by quality filters.
      </div>
    </>
  );
}

/* ────────────────────────────────────────────────────────────────────────
   Row 5 — MD-only target
   ──────────────────────────────────────────────────────────────────────── */

/**
 * Region balance against the MD-only target: extracted speech per region,
 * with MD highlighted (green) and every other region flagged as outside
 * the target. Uses the segment-level breakdown when available (it survives
 * rejects), otherwise the videos_master aggregate.
 */
export function RegionBalanceCard({ stats }) {
  const segRegions = (stats.segments || {}).by_region || {};
  const videoRegions = (stats.videos || {}).speech_by_region_s || {};

  // Merge both sources into region -> { durationS, clipCount? }.
  const regions = {};
  for (const [region, seconds] of Object.entries(videoRegions)) {
    regions[region] = { durationS: seconds, clipCount: null };
  }
  for (const [region, info] of Object.entries(segRegions)) {
    regions[region] = { durationS: info.duration_s || 0, clipCount: info.count || 0 };
  }

  const entries = Object.entries(regions).sort((a, b) => b[1].durationS - a[1].durationS);
  if (!entries.length) return <EmptyNote>No regional data yet</EmptyNote>;

  const totalS = entries.reduce((sum, [, r]) => sum + r.durationS, 0) || 1;
  const maxS = Math.max(1, ...entries.map(([, r]) => r.durationS));
  const targetS = regions[TARGET_REGION]?.durationS || 0;

  return (
    <>
      <RankList
        rows={entries.map(([region, r]) => {
          const isTarget = region === TARGET_REGION;
          const sharePct = (r.durationS / totalS) * 100;
          const clipInfo = r.clipCount != null ? ` · ${r.clipCount.toLocaleString()} clips` : '';
          return {
            key: region,
            label: (
              <>
                <i className={`dot ${isTarget ? 'green' : 'orange'}`}></i>
                {region}{isTarget ? ' (target)' : ' (outside target)'}
              </>
            ),
            value: `${formatDurationShort(r.durationS)} · ${sharePct.toFixed(1)}%${clipInfo}`,
            barPct: (r.durationS / maxS) * 100,
            barColor: isTarget ? 'var(--green)' : 'var(--orange)',
          };
        })}
      />
      <div className="progress-sub" style={{ marginTop: 6 }}>
        Dataset target is {TARGET_REGION}-only — {((targetS / totalS) * 100).toFixed(1)}% of
        extracted speech is on target; the rest will be excluded.
      </div>
    </>
  );
}

/**
 * Top speakers by total speaking time. Surfaces speaker imbalance early:
 * a corpus dominated by 2–3 speakers overfits the model to them.
 */
export function TopSpeakersCard({ speakers }) {
  if (!speakers.length) return <EmptyNote>No speakers yet</EmptyNote>;

  const totalDurS = speakers.reduce(
    (sum, sp) => sum + (parseFloat(sp.total_duration_s || 0) || 0), 0
  ) || 1;
  const top = [...speakers]
    .sort((a, b) => (parseFloat(b.total_duration_s || 0) || 0) - (parseFloat(a.total_duration_s || 0) || 0))
    .slice(0, 8);
  const maxDurS = Math.max(1, parseFloat(top[0]?.total_duration_s || 0) || 0);

  return (
    <>
      <RankList
        rows={top.map((sp) => {
          const durS = parseFloat(sp.total_duration_s || 0) || 0;
          const sharePct = (durS / totalDurS) * 100;
          return {
            key: sp.speaker_id,
            label: (
              <>
                <span className="rank-id">{sp.speaker_id}</span>
                {sp.speaker_name ? ` — ${sp.speaker_name}` : ''}
              </>
            ),
            value: `${formatDurationShort(durS)} · ${sharePct.toFixed(1)}%`,
            barPct: (durS / maxDurS) * 100,
          };
        })}
      />
      <div className="progress-sub" style={{ marginTop: 6 }}>
        Share of total speaking time across {speakers.length} speakers.
      </div>
    </>
  );
}

/* ────────────────────────────────────────────────────────────────────────
   The KPI strip above the grid
   ──────────────────────────────────────────────────────────────────────── */

/** A single KPI in the strip; `extra` renders under the sub-line. */
function Kpi({ value, label, sub, colorClass = '', groupEnd = false, primary = false, extra = null }) {
  return (
    <div className={`cmd-kpi ${groupEnd ? 'cmd-kpi-group-end' : ''} ${primary ? 'cmd-kpi-primary' : ''}`}>
      <span className={`cmd-kpi-value ${colorClass}`}>{value}</span>
      <span className="cmd-kpi-label">{label}</span>
      <span className="cmd-kpi-sub">{sub}</span>
      {extra}
    </div>
  );
}

/**
 * The strip of 7 KPIs: durations along the funnel (Source/Extracted/Approved),
 * clip states (Pending/Edited/Rejected), and the average WER.
 */
export function KpiStrip({ stats, actions = null }) {
  const v = stats.videos || {};
  const s = stats.segments || {};
  const ap = s.approved || {};
  const byStatus = v.by_status || {};

  const srcDurS = v.total_source_duration_s ?? (v.total_source_duration_h || 0) * 3600;
  const segDurS = s.total_duration_s ?? (s.total_duration_h || 0) * 3600;
  const apDurS = ap.total_duration_s || 0;

  const segTotal = s.total || 0;
  const approvedN = ap.count || 0;
  const rejectedN = s.rejected_count || 0;
  const pendingN = Math.max(0, segTotal - approvedN - rejectedN);
  const editedN = s.edited_count || 0;

  const pct = (num, denom) => (denom > 0 ? ((num / denom) * 100).toFixed(1) : '0.0');

  // The Source KPI's sub-line packs together all the video states.
  const sourceParts = [`${(v.total || 0).toLocaleString()} vids`];
  if (byStatus.completed) sourceParts.push(`${byStatus.completed} done`);
  if (byStatus.validated) sourceParts.push(`${byStatus.validated} validated`);
  if (byStatus.pending) sourceParts.push(`${byStatus.pending} pending`);
  if (byStatus.processing) sourceParts.push(`${byStatus.processing} processing`);
  if (byStatus.failed) sourceParts.push(`${byStatus.failed} failed`);

  // Progress toward the MD-only dataset target.
  const targetS = targetRegionSeconds(stats);
  const targetProgressPct = (targetS / (TARGET_HOURS * 3600)) * 100;

  return (
    <div className="cmd-kpi-strip">
      {/* Headline: progress toward the MD-only target, with the gauge line
          under the number — the strip's one deliberate accent. */}
      <Kpi
        value={`${(targetS / 3600).toFixed(1)}h / ${TARGET_HOURS}h`}
        label={`${TARGET_REGION} target`}
        colorClass="green"
        groupEnd
        primary
        sub={targetS
          ? `${targetProgressPct.toFixed(1)}% of the ${TARGET_REGION}-only dataset goal`
          : `no ${TARGET_REGION} speech extracted yet`}
        extra={
          <span className="kpi-target-gauge">
            <span
              className="kpi-target-gauge-fill"
              style={{ width: `${Math.min(100, targetProgressPct).toFixed(2)}%` }}
            />
          </span>
        }
      />

      {/* Training-ready: approved AND tier A/B — only once v3 data exists. */}
      {s.training_ready && (
        <Kpi
          value={formatDurationShort(s.training_ready.duration_s)}
          label="Training-ready"
          colorClass="green"
          sub={`${s.training_ready.count.toLocaleString()} clips · approved & tier A/B`}
        />
      )}

      {/* Group A: durations along the funnel */}
      <Kpi value={formatDurationShort(srcDurS)} label="Source video" sub={sourceParts.join(' · ')} />
      <Kpi
        value={formatDurationShort(segDurS)} label="Extracted audio" colorClass="cyan"
        sub={segTotal ? `${segTotal.toLocaleString()} clips · ${pct(segDurS, srcDurS)}% mined from source` : 'no clips yet'}
      />
      <Kpi
        value={formatDurationShort(apDurS)} label="Approved audio" colorClass="green" groupEnd
        sub={approvedN ? `${approvedN.toLocaleString()} clips · ${pct(apDurS, segDurS)}% of extracted` : 'none approved yet'}
      />

      {/* Group B: clip states */}
      <Kpi
        value={pendingN.toLocaleString()} label="Pending review" colorClass="orange"
        sub={segTotal ? `${pct(pendingN, segTotal)}% of clips waiting` : 'queue empty'}
      />
      <Kpi
        value={editedN.toLocaleString()} label="Edited segments" colorClass="cyan"
        sub={segTotal ? `${pct(editedN, segTotal)}% of clips · curator touch` : ''}
      />
      <Kpi
        value={rejectedN.toLocaleString()} label="Rejected" colorClass="red" groupEnd
        sub={rejectedN ? `${pct(rejectedN, segTotal)}% of clips · removed from disk` : 'none rejected'}
      />

      {/* Right-aligned actions (e.g. the data refresh button) */}
      {actions && <div className="cmd-kpi-actions">{actions}</div>}

      {/* The tail: edit quality */}
      <Kpi
        value={s.avg_wer == null ? '—' : `${(s.avg_wer * 100).toFixed(1)}%`}
        label="Avg WER"
        sub={
          s.avg_wer == null
            ? (s.with_wer_count ? `${s.with_wer_count.toLocaleString()} scored` : 'no WER yet')
            : `${(s.with_wer_count || 0).toLocaleString()} of ${segTotal.toLocaleString()} scored`
        }
      />
    </div>
  );
}

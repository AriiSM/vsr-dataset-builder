import { formatDurationShort } from '../../utils/format.js';
import { useStatsData } from './useStatsData.js';
import { ProgressChartCard } from './ProgressChartCard.jsx';
import {
  KpiStrip,
  PipelineFunnelCard, VideoStatusCard, QualityMetricsCard, HealthCard,
  DurationProfileCard, BySourceCard, TopChannelsCard, DemographicsCard,
  HistogramCard, ConfDistCard, VocabSummaryCard,
  TopVideosCard, WordStatsCard, DurationRangesCard,
  RegionBalanceCard, TopSpeakersCard, QualityTiersCard,
} from './insightsCards.jsx';

/**
 * A dashboard card. `span` stretches it across grid columns ('2') and
 * `tall` across two rows — the size encodes importance.
 */
function Card({ title, span, tall = false, children }) {
  const classes = [
    'overview-card',
    span === 2 ? 'card-span-2' : '',
    tall ? 'card-tall' : '',
  ].filter(Boolean).join(' ');
  return (
    <div className={classes}>
      <div className="stats-section-hdr">{title}</div>
      <div className="stats-section stats-scroll">{children}</div>
    </div>
  );
}

/**
 * The Dashboard tab — a bento grid whose hierarchy follows the curator's
 * questions in order:
 *   1. Am I on track for the 200h MD goal?   (big progress chart + region)
 *   2. What did the pipeline produce?        (funnel, tiers, statuses, health)
 *   3. What is the per-clip quality?         (distributions)
 *   4. Where does the corpus come from?      (speakers, sources, rankings)
 *   5. Reference numbers                     (compact tail row)
 */
export function DashboardTab({ isActive, refreshKey, onRefresh }) {
  const { stats, videos, distributions, speakers } = useStatsData(isActive, refreshKey);
  const videosStats = stats.videos || {};

  return (
    <main className={`tab-content ${isActive ? 'active' : ''}`} id="tab-dashboard">
      <section className="stats-sub-panel active">
        <KpiStrip
          stats={stats}
          actions={
            <button className="btn-micro" title="Reload data" onClick={onRefresh}>
              ↻ Refresh
            </button>
          }
        />

        <div className="dash-grid">
          {/* ── The headline: pace toward the MD-only goal ── */}
          <Card title={`Collection progress — toward 200h MD`} span={2} tall>
            <ProgressChartCard videos={videos} />
          </Card>
          <Card title="Region balance (MD target)">
            <RegionBalanceCard stats={stats} />
          </Card>
          <Card title="Pipeline flow"><PipelineFunnelCard stats={stats} /></Card>
          <Card title="Quality tiers (v3)"><QualityTiersCard stats={stats} /></Card>

          {/* ── Pipeline output & health ── */}
          <Card title="Videos by status"><VideoStatusCard stats={stats} /></Card>
          <Card title="Dataset health"><HealthCard distributions={distributions} /></Card>
          <Card title="Clip duration profile"><DurationProfileCard stats={stats} /></Card>
          <Card title="Confidence levels"><ConfDistCard stats={stats} /></Card>

          {/* ── Per-clip quality distributions ── */}
          <Card title="Segment duration distribution">
            <HistogramCard
              buckets={distributions?.duration_buckets || []}
              totalLabel="Segment duration (1s buckets)"
              axisLeft="1s"
              axisRight="15s"
              barClass="green"
            />
          </Card>
          <Card title="WER distribution">
            <HistogramCard
              buckets={distributions?.wer_buckets || []}
              totalLabel="WER (5% buckets)"
              axisLeft="0%"
              axisRight="100%"
              barClass="orange"
            />
          </Card>
          <Card title="Speaker demographics"><DemographicsCard speakers={speakers} /></Card>
          <Card title="Top speakers (speaking time)"><TopSpeakersCard speakers={speakers} /></Card>

          {/* ── Corpus provenance & rankings ── */}
          <Card title="By source / license"><BySourceCard stats={stats} /></Card>
          <Card title="Top channels"><TopChannelsCard stats={stats} /></Card>
          <Card title="Top videos (extracted duration)">
            <TopVideosCard
              rows={videosStats.top_extracted || []}
              getValue={(r) => ({ ref: r.extracted_s, label: formatDurationShort(r.extracted_s) })}
            />
          </Card>
          <Card title="Top videos (mining ratio)">
            <TopVideosCard
              rows={videosStats.top_mined || []}
              getValue={(r) => ({ ref: r.ratio_pct, label: `${r.ratio_pct.toFixed(1)}%` })}
            />
          </Card>

          {/* ── Reference numbers ── */}
          <Card title="Quality averages"><QualityMetricsCard stats={stats} /></Card>
          <Card title="Vocabulary coverage"><VocabSummaryCard stats={stats} /></Card>
          <Card title="Words & speech stats"><WordStatsCard stats={stats} /></Card>
          <Card title="Per-video duration ranges"><DurationRangesCard stats={stats} /></Card>
        </div>
      </section>
    </main>
  );
}

import { useStatsData } from './useStatsData.js';
import { ProgressChartCard } from './ProgressChartCard.jsx';
import {
  KpiStrip,
  PipelineFunnelCard, VideoStatusCard, QualityMetricsCard, HealthCard,
  QualityTiersCard, RegionBalanceCard,
  TrainingFunnelCard, ConsensusCard, SpeakerBalanceCard, LabelIssuesCard,
  TopSpeakersCard, DurationProfileCard, VocabSummaryCard,
} from './insightsCards.jsx';

/** A dashboard card, sized to its content. */
function Card({ title, children }) {
  return (
    <div className="overview-card">
      <div className="stats-section-hdr">{title}</div>
      <div className="stats-section stats-scroll">{children}</div>
    </div>
  );
}

/** Number of distinct regions in the segment + video breakdowns. */
function countRegions(stats) {
  return new Set([
    ...Object.keys((stats.segments || {}).by_region || {}),
    ...Object.keys((stats.videos || {}).speech_by_region_s || {}),
  ]).size;
}

export function DashboardTab({ isActive, refreshKey, onRefresh }) {
  const { stats, videos, distributions, speakers } = useStatsData(isActive, refreshKey);
  // With a single region (MD only) the balance card has nothing to compare.
  const showRegionBalance = countRegions(stats) > 1;

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

        {/* Two columns: the chart with the status cards under it, and the
            remaining cards packed beside it — no grid rows, so a short card
            never leaves a gap under itself. */}
        <div className="dash-grid">
          <div className="dash-col">
            <Card title="Collection progress — toward 200h MD">
              <ProgressChartCard videos={videos} />
            </Card>
            <div className="dash-pair">
              <Card title="Videos by status"><VideoStatusCard stats={stats} /></Card>
              <Card title="Dataset health"><HealthCard distributions={distributions} /></Card>
            </div>
          </div>
          <div className="dash-col dash-col-packed">
            <Card title="Pipeline flow"><PipelineFunnelCard stats={stats} /></Card>
            <Card title="Quality tiers"><QualityTiersCard stats={stats} /></Card>
            <Card title="Quality averages"><QualityMetricsCard stats={stats} /></Card>
            {showRegionBalance && (
              <Card title="Region balance (MD target)">
                <RegionBalanceCard stats={stats} />
              </Card>
            )}
          </div>

          {/* Second band: what the data looks like on the way to training. */}
          <div className="dash-band-title">Toward training</div>
          <div className="dash-col">
            <Card title="Training candidates — hours left after each filter">
              <TrainingFunnelCard stats={stats} />
            </Card>
            <div className="dash-pair">
              <Card title="Transcript consensus"><ConsensusCard stats={stats} /></Card>
              <Card title="Label issues"><LabelIssuesCard stats={stats} /></Card>
            </div>
          </div>
          <div className="dash-col dash-col-packed">
            <Card title="Speaker balance"><SpeakerBalanceCard stats={stats} /></Card>
            <Card title="Top speakers by speaking time"><TopSpeakersCard speakers={speakers} /></Card>
            <Card title="Clip durations"><DurationProfileCard stats={stats} /></Card>
            <Card title="Vocabulary"><VocabSummaryCard stats={stats} /></Card>
          </div>
        </div>
      </section>
    </main>
  );
}

import { useStatsData } from './useStatsData.js';
import { ProgressChartCard } from './ProgressChartCard.jsx';
import {
  VideoStatusCard, QualityMetricsCard, HealthCard,
  QualityTiersCard, RegionBalanceCard,
  TrainingFunnelCard, ConsensusCard, SpeakerBalanceCard, LabelIssuesCard,
  TopSpeakersCard, DurationProfileCard, VocabSummaryCard,
} from './insightsCards.jsx';
import { GoalHero, ReviewProgress, YieldCard } from './overviewHero.jsx';

/**
 * One dashboard panel. `span` is its width on the 12-column grid; `hint` is
 * a one-line explanation of how to read it.
 */
function Panel({ title, hint, span = 4, children }) {
  return (
    <section className={`ov-panel span-${span}`}>
      <div className="ov-panel-head">
        <h3>{title}</h3>
        {hint && <p>{hint}</p>}
      </div>
      <div className="ov-panel-body">{children}</div>
    </section>
  );
}

/** A titled group of panels. */
function Section({ title, lead, children }) {
  return (
    <div className="ov-section">
      <div className="ov-section-head">
        <h2>{title}</h2>
        {lead && <p>{lead}</p>}
      </div>
      <div className="ov-grid">{children}</div>
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

/**
 * The Overview: one question per section, read top to bottom.
 *   1. How far is the dataset from 200 h, at each quality level?
 *   2. Collection — is the pipeline still producing?
 *   3. Toward training — what survives the filters, and why the rest drops?
 *   4. Speakers & content — is the corpus balanced?
 *   5. Review — how much has a human checked?
 */
export function DashboardTab({ isActive, refreshKey, onRefresh }) {
  const { stats, videos, distributions, speakers } = useStatsData(isActive, refreshKey);
  // With a single region (MD only) the balance panel has nothing to compare.
  const showRegionBalance = countRegions(stats) > 1;

  return (
    <main className={`tab-content ${isActive ? 'active' : ''}`} id="tab-dashboard">
      <div className="ov-page">
        <div className="ov-page-head">
          <div>
            <h1>Overview</h1>
            <p>Moldovan-accented Romanian speech, from YouTube video to training-ready clips.</p>
          </div>
          <button className="btn-quiet" onClick={onRefresh} title="Reload all numbers">
            Refresh
          </button>
        </div>

        <GoalHero stats={stats} />

        <Section
          title="Collection"
          lead="How much video has been processed and how much speech it yielded."
        >
          <Panel title="Extracted speech over time" span={8}>
            <ProgressChartCard videos={videos} />
          </Panel>
          <div className="ov-stack span-4">
            <Panel title="Videos" span={12}><VideoStatusCard stats={stats} /></Panel>
            <Panel title="Yield" span={12}><YieldCard stats={stats} /></Panel>
          </div>
          {showRegionBalance && (
            <Panel title="Region balance" hint="Only MD counts toward the goal." span={12}>
              <RegionBalanceCard stats={stats} />
            </Panel>
          )}
        </Section>

        <Section
          title="Toward training"
          lead="Hours left after each automatic filter, and the reasons clips drop out."
        >
          <Panel
            title="Training candidates"
            hint="Each step keeps only the clips that passed every step above it."
            span={7}
          >
            <TrainingFunnelCard stats={stats} />
          </Panel>
          <Panel
            title="Transcript agreement"
            hint="Standard Whisper vs. the two dialect-prompted runs."
            span={5}
          >
            <ConsensusCard stats={stats} />
          </Panel>
          <Panel title="Quality tiers" hint="A and B are usable; C is flagged." span={4}>
            <QualityTiersCard stats={stats} />
          </Panel>
          <Panel title="Label issues" hint="Digits must be written as spoken words." span={4}>
            <LabelIssuesCard stats={stats} />
          </Panel>
          <Panel title="Signal averages" span={4}>
            <QualityMetricsCard stats={stats} />
            <div className="ov-divider" />
            <HealthCard distributions={distributions} />
          </Panel>
        </Section>

        <Section
          title="Speakers and content"
          lead="A corpus dominated by a few speakers makes a speaker-disjoint split hard."
        >
          <Panel title="Speaker balance" span={4}><SpeakerBalanceCard stats={stats} /></Panel>
          <Panel title="Top speakers by speaking time" span={8}>
            <TopSpeakersCard speakers={speakers} />
          </Panel>
          <Panel title="Clip durations" span={7}><DurationProfileCard stats={stats} /></Panel>
          <Panel title="Vocabulary" span={5}><VocabSummaryCard stats={stats} /></Panel>
        </Section>

        <Section title="Review" lead="Clips checked by hand in the Review page.">
          <ReviewProgress stats={stats} />
        </Section>
      </div>
    </main>
  );
}

import { useStatsData } from './useStatsData.js';
import { VideosTable } from './VideosTable.jsx';
import { SpeakersTable } from './SpeakersTable.jsx';
import { VocabTable } from './VocabTable.jsx';

/**
 * The Data tab: the raw registry tables — Videos (full width), Speakers and
 * Vocabulary side by side. Aggregated views live in the Dashboard tab.
 */
export function DataTab({ isActive, refreshKey, onRefresh }) {
  const { videos, speakers, vocabulary, reloadSpeakers } = useStatsData(isActive, refreshKey);

  return (
    <main className={`tab-content ${isActive ? 'active' : ''}`} id="tab-data">
      <section className="stats-sub-panel active">
        <div className="browse-layout">
          <VideosTable
            videos={videos}
            headerActions={
              <button className="btn-micro" title="Reload data" onClick={onRefresh}>
                ↻ Refresh
              </button>
            }
          />
          <div className="browse-row">
            <SpeakersTable speakers={speakers} onSpeakersChanged={reloadSpeakers} />
            <VocabTable words={vocabulary.words || []} totalUnique={vocabulary.total_unique} />
          </div>
        </div>
      </section>
    </main>
  );
}

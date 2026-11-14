import { useLocalStorage } from '../../hooks/useLocalStorage.js';
import { useStatsData } from './useStatsData.js';
import { VideosTable } from './VideosTable.jsx';
import { SpeakersTable } from './SpeakersTable.jsx';
import { VocabTable } from './VocabTable.jsx';

const VIEWS = [
  { id: 'videos',     label: 'Videos',     hint: 'Every source video, its status and what it yielded' },
  { id: 'speakers',   label: 'Speakers',   hint: 'One row per person, with demographics and speaking time' },
  { id: 'vocabulary', label: 'Vocabulary', hint: 'Every spoken word, with frequency and duration' },
];

/**
 * The Data tab: the raw registry tables, one at a time at full size.
 * All three stay mounted so search, sort and page survive switching.
 */
export function DataTab({ isActive, refreshKey, onRefresh }) {
  const { videos, speakers, vocabulary, reloadSpeakers } = useStatsData(isActive, refreshKey);
  const [stored, setView] = useLocalStorage('vsr-data-view', 'videos');
  const view = VIEWS.some((v) => v.id === stored) ? stored : 'videos';

  const counts = {
    videos: videos.length,
    speakers: speakers.length,
    vocabulary: vocabulary.total_unique || 0,
  };
  const active = VIEWS.find((v) => v.id === view);

  return (
    <main className={`tab-content ${isActive ? 'active' : ''}`} id="tab-data">
      <div className="data-head">
        <div className="data-head-text">
          <h1>Data</h1>
          <p>{active.hint}.</p>
        </div>
        <div className="data-switch" role="tablist" aria-label="Table">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              role="tab"
              aria-selected={view === v.id}
              className={`data-switch-btn ${view === v.id ? 'active' : ''}`}
              onClick={() => setView(v.id)}
            >
              {v.label}
              <span className="data-switch-count">{counts[v.id] ? counts[v.id].toLocaleString() : '…'}</span>
            </button>
          ))}
        </div>
        <button className="btn-quiet" onClick={onRefresh} title="Reload all tables">Refresh</button>
      </div>

      <div className="data-body">
        <div className={`data-view ${view === 'videos' ? 'active' : ''}`}>
          <VideosTable videos={videos} />
        </div>
        <div className={`data-view ${view === 'speakers' ? 'active' : ''}`}>
          <SpeakersTable speakers={speakers} onSpeakersChanged={reloadSpeakers} />
        </div>
        <div className={`data-view ${view === 'vocabulary' ? 'active' : ''}`}>
          <VocabTable words={vocabulary.words || []} totalUnique={vocabulary.total_unique} />
        </div>
      </div>
    </main>
  );
}

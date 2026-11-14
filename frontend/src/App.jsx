import { useState } from 'react';
import { useLocalStorage } from './hooks/useLocalStorage.js';
import { Toaster } from './components/toast.jsx';
import { ErrorBoundary } from './components/ErrorBoundary.jsx';
import { BackendStatus } from './components/BackendStatus.jsx';
import { ProcessTab } from './tabs/process/ProcessTab.jsx';
import { DashboardTab } from './tabs/stats/DashboardTab.jsx';
import { DataTab } from './tabs/stats/DataTab.jsx';
import { ExplorerTab } from './tabs/explorer/ExplorerTab.jsx';
import { ReviewTab } from './tabs/review/ReviewTab.jsx';

// Demo mode is set by the ?demo=1 URL parameter (read once — changing it
// requires a page reload anyway).
const IS_DEMO_MODE = new URLSearchParams(window.location.search).has('demo');

/**
 * The pages, grouped by what the user is doing: looking at the dataset,
 * curating it, or producing more of it.
 */
const NAV_GROUPS = [
  {
    label: 'Dataset',
    items: [
      { id: 'dashboard', label: 'Overview', icon: 'overview' },
      { id: 'data',      label: 'Data',     icon: 'table' },
      { id: 'explorer',  label: 'Explorer', icon: 'grid' },
    ],
  },
  {
    label: 'Curation',
    items: [{ id: 'review', label: 'Review', icon: 'check' }],
  },
  {
    label: 'Pipeline',
    items: [{ id: 'process', label: 'Process', icon: 'play' }],
  },
];
const TABS = NAV_GROUPS.flatMap((g) => g.items);
const TAB_IDS = TABS.map((t) => t.id);

/**
 * Maps a stored tab id to a valid one. The old single "stats" tab was split
 * into "dashboard" + "data", so sessions that saved 'stats' land on Dashboard.
 */
function normalizeTabId(id) {
  if (TAB_IDS.includes(id)) return id;
  if (id === 'stats') return 'dashboard';
  return 'process';
}

/**
 * Root component: the sidebar navigation + the five pages.
 *
 * All tabs stay permanently mounted (hidden via CSS) so their state is not
 * lost when switching — for example the pipeline polling keeps running even
 * while the user is looking at the Dashboard.
 */
export function App() {
  // The active tab persists across sessions ('stats' from older sessions is
  // migrated to 'dashboard').
  const [storedTab, setStoredTab] = useLocalStorage('vsr-active-tab', 'process');
  const activeTab = normalizeTabId(storedTab);
  // Incremented by the ↻ button — Dashboard/Data reload their data on change.
  const [refreshKey, setRefreshKey] = useState(0);

  return (
    <>
      <aside className="sidebar" aria-label="Main navigation">
        <div className="sidebar-brand">
          <span className="sidebar-mark" aria-hidden="true">VSR</span>
          <span className="sidebar-name">
            <strong>Dataset Builder</strong>
            <span>Romanian · Moldovan accent</span>
          </span>
        </div>

        <nav className="sidebar-nav">
          {NAV_GROUPS.map((group) => (
            <div className="sidebar-group" key={group.label}>
              <div className="sidebar-group-label">{group.label}</div>
              {group.items.map((tab) => (
                <button
                  key={tab.id}
                  className={`sidebar-item ${activeTab === tab.id ? 'active' : ''}`}
                  aria-current={activeTab === tab.id ? 'page' : undefined}
                  title={tab.label}
                  onClick={() => setStoredTab(tab.id)}
                >
                  <NavIcon name={tab.icon} />
                  <span className="sidebar-item-label">{tab.label}</span>
                </button>
              ))}
            </div>
          ))}
        </nav>

        <div className="sidebar-foot">
          {/* Persistent reminder that the numbers on screen are simulated. */}
          {IS_DEMO_MODE && (
            <span className="demo-badge" title="Demo mode — replaying a simulated log. Remove ?demo=1 from the URL to exit.">
              DEMO
            </span>
          )}
          {/* Live backend indicator — pointless in demo mode (no backend). */}
          {!IS_DEMO_MODE && <BackendStatus />}
        </div>
      </aside>

      <div className="app-main">

      {/* All tabs are mounted; only the active one is visible (CSS). Every
          tab gets its own error boundary so a render crash in one tab
          cannot take down the whole app. */}
      <ErrorBoundary label="Process" active={activeTab === 'process'}>
        <ProcessTab
          isActive={activeTab === 'process'}
          onPipelineRunning={() => setStoredTab('process')}
        />
      </ErrorBoundary>
      <ErrorBoundary label="Review" active={activeTab === 'review'}>
        <ReviewTab isActive={activeTab === 'review'} />
      </ErrorBoundary>
      <ErrorBoundary label="Explorer" active={activeTab === 'explorer'}>
        <ExplorerTab isActive={activeTab === 'explorer'} />
      </ErrorBoundary>
      <ErrorBoundary label="Dashboard" active={activeTab === 'dashboard'}>
        <DashboardTab
          isActive={activeTab === 'dashboard'}
          refreshKey={refreshKey}
          onRefresh={() => setRefreshKey((k) => k + 1)}
        />
      </ErrorBoundary>
      <ErrorBoundary label="Data" active={activeTab === 'data'}>
        <DataTab
          isActive={activeTab === 'data'}
          refreshKey={refreshKey}
          onRefresh={() => setRefreshKey((k) => k + 1)}
        />
      </ErrorBoundary>

      </div>

      {/* Toast notifications (bottom-right corner) */}
      <Toaster />
    </>
  );
}

/** Small line icons for the sidebar (inline SVG, inherit currentColor). */
function NavIcon({ name }) {
  const paths = {
    overview: <><path d="M4 19V11" /><path d="M10 19V5" /><path d="M16 19v-6" /><path d="M3 19h18" /></>,
    table: <><rect x="3.5" y="4.5" width="17" height="15" rx="2" /><path d="M3.5 9.5h17M3.5 14.5h17M9.5 9.5v10" /></>,
    grid: <><rect x="4" y="4" width="7" height="7" rx="1.5" /><rect x="13" y="4" width="7" height="7" rx="1.5" /><rect x="4" y="13" width="7" height="7" rx="1.5" /><rect x="13" y="13" width="7" height="7" rx="1.5" /></>,
    check: <><path d="M20 12a8 8 0 1 1-4.7-7.3" /><path d="M9 11.5l3 3 8-8" /></>,
    play: <><circle cx="12" cy="12" r="8.5" /><path d="M10 8.8v6.4l5.2-3.2z" /></>,
  };
  return (
    <svg className="sidebar-icon" viewBox="0 0 24 24" width="18" height="18" fill="none"
      stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {paths[name]}
    </svg>
  );
}

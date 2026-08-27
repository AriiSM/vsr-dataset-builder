import { useLayoutEffect, useRef, useState } from 'react';
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
 * The main tabs, in two visual groups that mirror the two kinds of work:
 * running & curating the pipeline vs. exploring & analysing the dataset.
 */
const TAB_GROUPS = [
  [
    { id: 'process',   label: 'Process' },
    { id: 'review',    label: 'Review' },
  ],
  [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'data',      label: 'Data' },
    { id: 'explorer',  label: 'Explorer' },
  ],
];
const TAB_IDS = TAB_GROUPS.flat().map((t) => t.id);

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
 * Root component: the header with tabs + the five pages.
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

  // ── Sliding nav indicator ────────────────────────────────────────────────
  // One cyan bar that glides under the active tab instead of per-tab
  // underlines: measured from the active button, re-measured on resize.
  const navRef = useRef(null);
  const [indicator, setIndicator] = useState({ left: 0, width: 0 });

  useLayoutEffect(() => {
    const measure = () => {
      const nav = navRef.current;
      const activeButton = nav?.querySelector('.tab.active');
      if (nav && activeButton) {
        // Rect-based (not offsetLeft): the buttons sit inside group pills,
        // so offsets must be relative to the whole nav.
        const navRect = nav.getBoundingClientRect();
        const btnRect = activeButton.getBoundingClientRect();
        setIndicator({
          left: btnRect.left - navRect.left + 10,
          width: btnRect.width - 20,
        });
      }
    };
    measure();
    window.addEventListener('resize', measure);
    return () => window.removeEventListener('resize', measure);
  }, [activeTab]);

  return (
    <>
      <header>
        <div className="header-left">
          <div className="logo">VSR</div>
          <span className="header-sep"></span>
          <span className="header-subtitle">Romanian VSR · Dataset Builder</span>
          {/* Persistent reminder that the numbers on screen are simulated. */}
          {IS_DEMO_MODE && (
            <span className="demo-badge" title="Demo mode — replaying a simulated log. Remove ?demo=1 from the URL to exit.">
              DEMO
            </span>
          )}
          {/* Live backend indicator — pointless in demo mode (no backend). */}
          {!IS_DEMO_MODE && <BackendStatus />}
        </div>
        <div className="header-right">
          <nav className="tabs" ref={navRef}>
            {TAB_GROUPS.map((group, groupIndex) => (
              <div className="tab-group" key={groupIndex}>
                {group.map((tab) => (
                  <button
                    key={tab.id}
                    className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                    onClick={() => setStoredTab(tab.id)}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>
            ))}
            <span
              className="nav-indicator"
              style={{ transform: `translateX(${indicator.left}px)`, width: indicator.width }}
            />
          </nav>
        </div>
      </header>

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

      {/* Toast notifications (bottom-right corner) */}
      <Toaster />
    </>
  );
}

import React from 'react';

/**
 * Error boundary that isolates render crashes to one tab: if a component
 * inside throws (e.g. a Stats card on unexpected data), only that tab shows
 * an error message — the rest of the app keeps working.
 *
 * Must be a class component: React only supports componentDidCatch /
 * getDerivedStateFromError on classes.
 */
export class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error(`[ErrorBoundary:${this.props.label || 'tab'}]`, error, info);
  }

  render() {
    if (this.state.error) {
      // The fallback replaces the tab's own <main>, so it must follow the
      // same visibility rule — only the active tab's fallback is shown.
      return (
        <main
          className={`tab-content ${this.props.active ? 'active' : ''}`}
          style={{ padding: 32 }}
        >
          <div
            style={{
              maxWidth: 560, margin: '48px auto', padding: 24,
              background: 'var(--surface)', border: '1px solid var(--border)',
              borderLeft: '3px solid var(--red)', borderRadius: 'var(--radius)',
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 8 }}>
              Something went wrong in the {this.props.label || 'current'} tab
            </div>
            <pre
              style={{
                fontFamily: 'var(--font-mono)', fontSize: 11,
                color: 'var(--text-dim)', whiteSpace: 'pre-wrap', marginBottom: 12,
              }}
            >
              {String(this.state.error?.message || this.state.error)}
            </pre>
            <button className="btn-micro" onClick={() => this.setState({ error: null })}>
              Try again
            </button>
          </div>
        </main>
      );
    }
    return this.props.children;
  }
}

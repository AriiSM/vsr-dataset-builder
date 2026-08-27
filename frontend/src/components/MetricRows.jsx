/**
 * Simple list of "label — value" rows, used by several cards
 * in the Stats tab (quality metrics, vocabulary, durations).
 *
 * rows: [{ label: string, value: string }]
 */
export function MetricRows({ rows }) {
  return (
    <>
      {rows.map((row) => (
        <div className="metric-row" key={row.label}>
          <span className="metric-label">{row.label}</span>
          <span className="metric-value mono">{row.value}</span>
        </div>
      ))}
    </>
  );
}

/** Standard message for cards with no data yet. */
export function EmptyNote({ children }) {
  return <div className="dim" style={{ padding: '4px 0' }}>{children}</div>;
}

/** Sub-section heading inside a card (e.g. "By length band"). */
export function CardSubheading({ children }) {
  return (
    <div
      className="metric-row"
      style={{ border: 'none', padding: '8px 0 4px', color: 'var(--text-secondary)', fontWeight: 700 }}
    >
      {children}
    </div>
  );
}

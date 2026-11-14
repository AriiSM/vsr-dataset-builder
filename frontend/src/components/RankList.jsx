/**
 * Ranking list with horizontal bars — used for "top channels",
 * "top videos", distributions by region/source, etc.
 *
 * rows: [{ key, label (JSX or text), value (text), barPct (0–100), barColor? }]
 */
export function RankList({ rows }) {
  return (
    <>
      {rows.map((row) => (
        <div className="rank-row" key={row.key}>
          <span className="rank-label">{row.label}</span>
          <span className="rank-value">{row.value}</span>
          <div className="rank-bar-wrap">
            <div
              className="rank-bar-fill"
              style={{
                width: `${row.barPct.toFixed(1)}%`,
                ...(row.barColor ? { background: row.barColor } : {}),
              }}
            />
          </div>
        </div>
      ))}
    </>
  );
}

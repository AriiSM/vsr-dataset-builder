import { useMemo, useState } from 'react';
import { TARGET_REGION, TARGET_HOURS } from '../../config.js';
import { EmptyNote } from '../../components/MetricRows.jsx';

/**
 * Collection progress toward the MD-only target: a cumulative line of
 * extracted MD hours over processing days, with the 200h goal drawn as a
 * reference line. Answers the dashboard's headline question — "at this
 * pace, am I on track?".
 *
 * Chart notes (per the dataviz method): single series → no legend, the
 * title names it; brand cyan for the series, status green for the goal
 * line; recessive grid; hover crosshair + tooltip; all text in text tokens.
 */

const WIDTH = 640;
const HEIGHT = 240;
const PAD = { top: 16, right: 16, bottom: 26, left: 44 };

/** Builds cumulative [{dayMs, label, hours}] from the video rows. */
function buildSeries(videos) {
  const perDay = new Map();
  for (const video of videos) {
    if (String(video.region || '').trim() !== TARGET_REGION) continue;
    const durS = parseFloat(video.total_duration_extracted);
    if (!Number.isFinite(durS) || durS <= 0) continue;
    // processed_date may be "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS".
    const day = String(video.processed_date || '').slice(0, 10);
    if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) continue;
    perDay.set(day, (perDay.get(day) || 0) + durS);
  }
  const days = [...perDay.keys()].sort();
  let cumulativeS = 0;
  return days.map((day) => {
    cumulativeS += perDay.get(day);
    return { dayMs: Date.parse(day), label: day, hours: cumulativeS / 3600 };
  });
}

export function ProgressChartCard({ videos }) {
  const points = useMemo(() => buildSeries(videos), [videos]);
  const [hoverIndex, setHoverIndex] = useState(null);

  if (points.length < 2) {
    return (
      <EmptyNote>
        Not enough dated {TARGET_REGION} processing history yet — the curve
        appears once at least two processing days exist.
      </EmptyNote>
    );
  }

  // ── Scales ──
  const innerW = WIDTH - PAD.left - PAD.right;
  const innerH = HEIGHT - PAD.top - PAD.bottom;
  const minMs = points[0].dayMs;
  const maxMs = points[points.length - 1].dayMs;
  const maxHours = Math.max(TARGET_HOURS, points[points.length - 1].hours) * 1.05;
  const x = (ms) => PAD.left + ((ms - minMs) / Math.max(1, maxMs - minMs)) * innerW;
  const y = (hours) => PAD.top + innerH - (hours / maxHours) * innerH;

  const linePath = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${x(p.dayMs).toFixed(1)},${y(p.hours).toFixed(1)}`)
    .join(' ');
  const areaPath =
    `${linePath} L${x(maxMs).toFixed(1)},${(PAD.top + innerH).toFixed(1)}` +
    ` L${x(minMs).toFixed(1)},${(PAD.top + innerH).toFixed(1)} Z`;

  // Recessive y grid: a tick roughly every 50h, skipping 0.
  const yTicks = [];
  for (let h = 50; h < maxHours; h += 50) yTicks.push(h);

  /** Snap the mouse to the nearest data point (bigger hit target than the dot). */
  function handleMove(event) {
    const svgRect = event.currentTarget.getBoundingClientRect();
    const mouseMs =
      minMs + ((event.clientX - svgRect.left) / svgRect.width * WIDTH - PAD.left)
        / innerW * (maxMs - minMs);
    let nearest = 0;
    for (let i = 1; i < points.length; i++) {
      if (Math.abs(points[i].dayMs - mouseMs) < Math.abs(points[nearest].dayMs - mouseMs)) {
        nearest = i;
      }
    }
    setHoverIndex(nearest);
  }

  const hover = hoverIndex != null ? points[hoverIndex] : null;
  const latest = points[points.length - 1];
  const middle = points[Math.floor(points.length / 2)];

  return (
    <div className="progress-chart-wrap">
      <svg
        className="progress-chart"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`Cumulative ${TARGET_REGION} hours over time, toward the ${TARGET_HOURS}h goal`}
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverIndex(null)}
      >
        {/* Recessive grid + y labels */}
        {yTicks.map((h) => (
          <g key={h}>
            <line x1={PAD.left} x2={WIDTH - PAD.right} y1={y(h)} y2={y(h)} className="chart-grid" />
            <text x={PAD.left - 6} y={y(h) + 3} className="chart-tick" textAnchor="end">{h}h</text>
          </g>
        ))}

        {/* Goal line: status green, dashed, labeled */}
        <line
          x1={PAD.left} x2={WIDTH - PAD.right}
          y1={y(TARGET_HOURS)} y2={y(TARGET_HOURS)}
          className="chart-goal"
        />
        <text x={WIDTH - PAD.right} y={y(TARGET_HOURS) - 5} className="chart-goal-label" textAnchor="end">
          goal {TARGET_HOURS}h
        </text>

        {/* Series: soft area + 2px line, brand cyan */}
        <path d={areaPath} className="chart-area" />
        <path d={linePath} className="chart-line" />

        {/* Latest value, directly labeled (selective labeling) */}
        <circle cx={x(latest.dayMs)} cy={y(latest.hours)} r="3.5" className="chart-dot" />
        <text
          x={Math.min(x(latest.dayMs), WIDTH - PAD.right - 4)}
          y={y(latest.hours) - 8}
          className="chart-value-label"
          textAnchor="end"
        >
          {latest.hours.toFixed(1)}h
        </text>

        {/* x labels: first / middle / last day */}
        {[points[0], middle, latest].map((p, i) => (
          <text
            key={i}
            x={x(p.dayMs)}
            y={HEIGHT - 8}
            className="chart-tick"
            textAnchor={i === 0 ? 'start' : i === 2 ? 'end' : 'middle'}
          >
            {p.label.slice(5)}
          </text>
        ))}

        {/* Hover: crosshair + emphasized point */}
        {hover && (
          <g>
            <line
              x1={x(hover.dayMs)} x2={x(hover.dayMs)}
              y1={PAD.top} y2={PAD.top + innerH}
              className="chart-crosshair"
            />
            <circle cx={x(hover.dayMs)} cy={y(hover.hours)} r="4.5" className="chart-dot-hover" />
          </g>
        )}
      </svg>

      {/* Tooltip (HTML, under the plot so it never collides with marks) */}
      <div className="chart-readout">
        {hover ? (
          <>
            <span className="mono">{hover.label}</span>
            <span className="mono chart-readout-value">{hover.hours.toFixed(1)}h</span>
            <span className="dim">{((hover.hours / TARGET_HOURS) * 100).toFixed(1)}% of goal</span>
          </>
        ) : (
          <>
            <span className="dim">{points.length} processing days</span>
            <span className="mono chart-readout-value">{latest.hours.toFixed(1)}h</span>
            <span className="dim">{((latest.hours / TARGET_HOURS) * 100).toFixed(1)}% of the {TARGET_HOURS}h goal</span>
          </>
        )}
      </div>
    </div>
  );
}

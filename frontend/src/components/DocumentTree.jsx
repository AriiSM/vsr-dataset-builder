import { useState } from 'react';

/**
 * Renders a MongoDB document the way it is stored: sub-documents as nested,
 * collapsible groups, arrays of sub-documents (words[], dropped_clips[]) as
 * compact tables, scalars as key/value rows. Dates arrive as ISO strings.
 */

const ISO_DATE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/;

function formatScalar(value) {
  if (value === null || value === undefined) return <span className="doc-null">null</span>;
  if (typeof value === 'boolean') return <span className="doc-bool">{String(value)}</span>;
  if (typeof value === 'number') {
    const text = Number.isInteger(value) ? String(value) : String(+value.toFixed(4));
    return <span className="doc-num">{text}</span>;
  }
  if (typeof value === 'string' && ISO_DATE.test(value)) {
    return <span className="doc-date">{value.replace('T', ' ').slice(0, 19)}</span>;
  }
  return <span className="doc-str">{String(value)}</span>;
}

const isPlainObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

function ArrayTable({ items, limit }) {
  const [showAll, setShowAll] = useState(false);
  const columns = [];
  const flatten = (obj, prefix = '') => {
    const out = {};
    Object.entries(obj).forEach(([k, v]) => {
      const key = prefix ? `${prefix}.${k}` : k;
      if (isPlainObject(v)) Object.assign(out, flatten(v, key));
      else out[key] = v;
    });
    return out;
  };
  const rows = items.map((item) => flatten(item));
  rows.forEach((r) => Object.keys(r).forEach((k) => { if (!columns.includes(k)) columns.push(k); }));
  const visible = showAll ? rows : rows.slice(0, limit);
  return (
    <div className="doc-array-wrap">
      <table className="doc-array">
        <thead>
          <tr><th>#</th>{columns.map((c) => <th key={c}>{c}</th>)}</tr>
        </thead>
        <tbody>
          {visible.map((r, i) => (
            <tr key={i}>
              <td className="doc-idx">{i}</td>
              {columns.map((c) => <td key={c}>{c in r ? formatScalar(r[c]) : ''}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > limit && (
        <button className="btn-micro" onClick={() => setShowAll((s) => !s)}>
          {showAll ? 'Show less' : `Show all ${rows.length}`}
        </button>
      )}
    </div>
  );
}

function Node({ name, value, depth, arrayLimit }) {
  const [open, setOpen] = useState(depth < 2);
  if (Array.isArray(value)) {
    const objects = value.length > 0 && value.every(isPlainObject);
    return (
      <div className="doc-node">
        <button className="doc-key doc-toggle" onClick={() => setOpen((o) => !o)}>
          {open ? '▾' : '▸'} {name} <span className="doc-meta">[{value.length}]</span>
        </button>
        {open && (objects
          ? <ArrayTable items={value} limit={arrayLimit} />
          : <div className="doc-children">{value.map((v, i) => (
              <Node key={i} name={i} value={v} depth={depth + 1} arrayLimit={arrayLimit} />))}</div>)}
      </div>
    );
  }
  if (isPlainObject(value)) {
    const entries = Object.entries(value);
    return (
      <div className="doc-node">
        <button className="doc-key doc-toggle" onClick={() => setOpen((o) => !o)}>
          {open ? '▾' : '▸'} {name} <span className="doc-meta">{'{'}{entries.length}{'}'}</span>
        </button>
        {open && (
          <div className="doc-children">
            {entries.map(([k, v]) => (
              <Node key={k} name={k} value={v} depth={depth + 1} arrayLimit={arrayLimit} />
            ))}
          </div>
        )}
      </div>
    );
  }
  return (
    <div className="doc-row">
      <span className="doc-key">{name}</span>
      {formatScalar(value)}
    </div>
  );
}

export function DocumentTree({ document, arrayLimit = 20 }) {
  if (!document) return null;
  return (
    <div className="doc-tree">
      {Object.entries(document).map(([k, v]) => (
        <Node key={k} name={k} value={v} depth={0} arrayLimit={arrayLimit} />
      ))}
    </div>
  );
}

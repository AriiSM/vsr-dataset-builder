import { paginationRange } from '../utils/format.js';

/**
 * The pagination bar used by all tables: ‹ 1 … 4 [5] 6 … 20 ›
 * Renders nothing if there is only one page.
 */
export function Pagination({ page, totalPages, onPageChange }) {
  if (totalPages <= 1) return null;

  return (
    <div className="pagination">
      <button
        className="page-btn"
        onClick={() => onPageChange(page - 1)}
        disabled={page === 1}
      >
        ‹
      </button>
      {paginationRange(page, totalPages).map((p, i) =>
        p === '...' ? (
          <span key={`gap-${i}`} className="page-ellipsis">...</span>
        ) : (
          <button
            key={p}
            className={`page-btn ${p === page ? 'active' : ''}`}
            onClick={() => onPageChange(p)}
          >
            {p}
          </button>
        )
      )}
      <button
        className="page-btn"
        onClick={() => onPageChange(page + 1)}
        disabled={page === totalPages}
      >
        ›
      </button>
    </div>
  );
}

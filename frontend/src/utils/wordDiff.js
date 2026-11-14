/**
 * Word-level diff between two short texts (LCS-based).
 *
 * Used to show what the large-v3 transcript refiner would change relative
 * to the current transcript. Texts are sentence-sized (tens of words), so
 * the O(n·m) dynamic programming table is negligible.
 *
 * Returns ops for the SUGGESTED text: each word tagged with
 *   'same'    — word kept from the current text
 *   'changed' — word added/replaced by the suggestion
 */
export function diffWords(currentText, suggestedText) {
  const a = String(currentText || '').trim().split(/\s+/).filter(Boolean);
  const b = String(suggestedText || '').trim().split(/\s+/).filter(Boolean);

  // LCS length table (case-insensitive comparison, punctuation kept).
  const eq = (x, y) => x.toLowerCase() === y.toLowerCase();
  const table = Array.from({ length: a.length + 1 }, () =>
    new Array(b.length + 1).fill(0)
  );
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      table[i][j] = eq(a[i], b[j])
        ? table[i + 1][j + 1] + 1
        : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }

  // Walk the table and tag each word of the suggestion.
  const ops = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (eq(a[i], b[j])) {
      ops.push({ type: 'same', word: b[j] });
      i++;
      j++;
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      i++; // word deleted from the current text — not shown in suggestion
    } else {
      ops.push({ type: 'changed', word: b[j] });
      j++;
    }
  }
  for (; j < b.length; j++) {
    ops.push({ type: 'changed', word: b[j] });
  }
  return ops;
}

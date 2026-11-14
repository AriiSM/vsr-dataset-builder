"""Plot a histogram of segment duration distribution for the MD VSR dataset.

Reads `data/metadata/segments_index.csv` and bins segment durations.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BIN_EDGES = [1, 3, 5, 7, 10, 15]
BIN_LABELS = ["1-3 s", "3-5 s", "5-7 s", "7-10 s", "10-15 s"]

BAR_COLOR = "#2E83A0"
TEXT_COLOR = "#222222"


def load_durations(csv_path: Path) -> list[float]:
    durations = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("duration", "").strip()
            if not raw:
                continue
            try:
                durations.append(float(raw))
            except ValueError:
                continue
    return durations


def bin_durations(durations: list[float]) -> list[int]:
    counts = [0] * len(BIN_LABELS)
    for d in durations:
        for i in range(len(BIN_LABELS)):
            lo, hi = BIN_EDGES[i], BIN_EDGES[i + 1]
            # Last bin is inclusive on the right; others are [lo, hi).
            in_bin = (lo <= d < hi) if i < len(BIN_LABELS) - 1 else (lo <= d <= hi)
            if in_bin:
                counts[i] += 1
                break
    return counts


def plot(counts: list[int], title: str, output_path: Path | None = None) -> None:
    total = sum(counts)
    if total == 0:
        raise SystemExit("No segments found in the given duration range.")
    percentages = [c / total * 100 for c in counts]

    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(9, 6.5), facecolor="white")
    ax.set_facecolor("white")

    x = np.arange(len(BIN_LABELS))
    bars = ax.bar(x, counts, color=BAR_COLOR, width=0.62, edgecolor="white", linewidth=0.5)

    max_count = max(counts)
    count_offset = max_count * 0.018
    pct_offset = max_count * 0.062

    for bar, count, pct in zip(bars, counts, percentages):
        cx = bar.get_x() + bar.get_width() / 2
        h = bar.get_height()
        ax.text(
            cx, h + count_offset,
            f"{count:,}",
            ha="center", va="bottom", fontsize=11, color=TEXT_COLOR, fontweight="bold",
        )
        ax.text(
            cx, h + pct_offset,
            f"({pct:.1f}%)",
            ha="center", va="bottom", fontsize=9, color="#666666",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, fontsize=11, color="#333333")
    ax.tick_params(axis="y", labelsize=10, colors="#333333")
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Number of segments", fontsize=11, color="#333333")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=18, color="#1A1A1A")

    ax.set_ylim(0, max_count * 1.22)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v):,}"))

    ax.grid(False)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#BBBBBB")
        ax.spines[spine].set_linewidth(0.8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Saved histogram to: {output_path}")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MD VSR segment duration histogram.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("data/metadata/segments_index.csv"),
        help="Path to segments_index.csv (default: data/metadata/segments_index.csv).",
    )
    parser.add_argument(
        "--title",
        default="MD Dataset — Segment Duration Distribution",
        help="Figure title.",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Optional path to save the figure (e.g. duration_hist.png).")
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"CSV not found: {args.csv}")

    durations = load_durations(args.csv)
    counts = bin_durations(durations)

    total_in_range = sum(counts)
    total_loaded = len(durations)
    out_of_range = total_loaded - total_in_range

    print(f"Loaded {total_loaded:,} segments from {args.csv}")
    print(f"In range [{BIN_EDGES[0]}s, {BIN_EDGES[-1]}s]: {total_in_range:,}")
    if out_of_range > 0:
        print(f"Outside range (skipped): {out_of_range:,}")
    for label, c in zip(BIN_LABELS, counts):
        pct = c / total_in_range * 100 if total_in_range else 0.0
        print(f"  {label:>8}: {c:>7,}  ({pct:5.1f}%)")

    plot(counts, args.title, args.output)


if __name__ == "__main__":
    main()

"""Compute corpus metadata statistics for the MD VSR dataset.

Reads:
  - data/metadata/videos_master.csv         (source, source_channel)
  - data/metadata/speakers_registry.csv     (gender, age_group, duration)
  - data/metadata/segments_index.csv        (interaction type via distinct speaker_id per video)
  - data/metadata/auto_tag_workdir/proposals.csv  (auto-tagged environment + noise)

Prints both human-readable stats and a LaTeX block ready to paste into the
"Corpus metadata overview" table.
"""

import argparse
import collections
import csv
from pathlib import Path

DATA_DIR_DEFAULT = Path("data/metadata")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pct(n: int, d: int) -> float:
    return (n / d * 100) if d else 0.0


def env_stats(proposals: list[dict]) -> dict:
    """Auto-tag: only 'studio' vs unlabeled (treated as Indoor)."""
    studio = sum(1 for p in proposals if p.get("channel_env_prior", "").strip() == "studio")
    total = len(proposals)
    indoor = total - studio  # treat unlabeled as indoor (TV interviews → no outdoor)
    return {
        "Studio": pct(studio, total),
        "Indoor": pct(indoor, total),
        "Outdoor": 0.0,
        "_total": total,
        "_note": "from auto-tag (channel_env_prior), not manual",
    }


def noise_stats(proposals: list[dict]) -> dict:
    """Auto-tag: merge 'none'+'low' → Low; 'moderate' → Medium; 'high' → High."""
    c = collections.Counter(p.get("suggested_noise", "").strip() for p in proposals)
    low = c["low"] + c["none"]
    medium = c["moderate"]
    high = c["high"]
    known = low + medium + high
    return {
        "Low": pct(low, known),
        "Medium": pct(medium, known),
        "High": pct(high, known),
        "_total": known,
        "_note": "from auto-tag (RMS + spectral flatness)",
    }


def source_stats(videos: list[dict]) -> dict:
    c = collections.Counter(v.get("source_channel", "").strip() for v in videos)
    top_name, top_count = c.most_common(1)[0]
    total = len(videos)
    return {
        "Top": pct(top_count, total),
        "Other": pct(total - top_count, total),
        "_top_name": top_name,
        "_total": total,
    }


def gender_stats(speakers: list[dict], duration_weighted: bool = False) -> dict:
    if duration_weighted:
        acc = collections.defaultdict(float)
        for s in speakers:
            g = s.get("gender", "").strip()
            if g not in ("M", "F"):
                continue
            try:
                d = float(s.get("total_duration_s", "0") or 0)
            except ValueError:
                d = 0
            acc[g] += d
        tot = acc["M"] + acc["F"]
        return {"Male": pct(acc["M"], tot), "Female": pct(acc["F"], tot), "_total": tot, "_weighting": "duration"}
    c = collections.Counter(s.get("gender", "").strip() for s in speakers)
    known = c["M"] + c["F"]
    return {"Male": pct(c["M"], known), "Female": pct(c["F"], known), "_total": known, "_weighting": "speaker-count"}


def age_stats(speakers: list[dict], duration_weighted: bool = False) -> dict:
    bins = ("18-30", "31-50", "51+")
    if duration_weighted:
        acc = collections.defaultdict(float)
        for s in speakers:
            a = s.get("age_group", "").strip()
            if a not in bins:
                continue
            try:
                d = float(s.get("total_duration_s", "0") or 0)
            except ValueError:
                d = 0
            acc[a] += d
        tot = sum(acc.values())
        return {b: pct(acc[b], tot) for b in bins} | {"_total": tot, "_weighting": "duration"}
    c = collections.Counter(s.get("age_group", "").strip() for s in speakers)
    known = sum(c[b] for b in bins)
    return {b: pct(c[b], known) for b in bins} | {"_total": known, "_weighting": "speaker-count"}


def interaction_stats(segments: list[dict]) -> dict:
    spk_per_vid = collections.defaultdict(set)
    for r in segments:
        spk_per_vid[r["video_id"]].add(r.get("speaker_id", "").strip())
    buckets = collections.Counter()
    for vid, sp in spk_per_vid.items():
        sp.discard("")
        n = len(sp)
        if n == 1:
            buckets["monologue"] += 1
        elif n == 2:
            buckets["interview"] += 1
        elif n >= 3:
            buckets["other"] += 1
    tot = sum(buckets.values())
    return {
        "Monologue": pct(buckets["monologue"], tot),
        "Interview": pct(buckets["interview"], tot),
        "Other": pct(buckets["other"], tot),
        "_total": tot,
    }


def print_section(title: str, items: list[tuple[str, float]], note: str = "") -> None:
    print(f"\n{title}" + (f"  ({note})" if note else ""))
    for label, value in items:
        print(f"  {label:<28s} {value:5.1f}%")


def print_latex(env, noise, src, gender, age, inter, ro: dict) -> None:
    print("\n" + "=" * 60)
    print("LaTeX block (paste into the table):")
    print("=" * 60)
    rows = [
        ("\\multicolumn{3}{l}{\\textit{Recording environment}} \\\\", None),
        (f"Studio recordings   & {ro['studio_ro']}\\% & {env['Studio']:.1f}\\% \\\\", None),
        (f"Indoor recordings   & {ro['indoor_ro']}\\% & {env['Indoor']:.1f}\\% \\\\", None),
        (f"Outdoor recordings  & {ro['outdoor_ro']}\\% & {env['Outdoor']:.1f}\\% \\\\", None),
        ("\\midrule", None),
        ("\\multicolumn{3}{l}{\\textit{Audio conditions}} \\\\", None),
        (f"Low background noise    & {ro['noise_low']}\\% & {noise['Low']:.1f}\\% \\\\", None),
        (f"Medium background noise & {ro['noise_med']}\\% & {noise['Medium']:.1f}\\% \\\\", None),
        (f"High background noise   & {ro['noise_high']}\\% & {noise['High']:.1f}\\% \\\\", None),
        ("\\midrule", None),
        ("\\multicolumn{3}{l}{\\textit{Source diversity}} \\\\", None),
        (f"Top source share        & {ro['top_share']}\\% & {src['Top']:.1f}\\% \\\\", None),
        (f"Other sources           & {ro['other_share']}\\% & {src['Other']:.1f}\\% \\\\", None),
        ("\\midrule", None),
        ("\\multicolumn{3}{l}{\\textit{Gender distribution}} \\\\", None),
        (f"Male speakers        & {ro['male']}\\% & {gender['Male']:.1f}\\% \\\\", None),
        (f"Female speakers      & {ro['female']}\\% & {gender['Female']:.1f}\\% \\\\", None),
        ("\\midrule", None),
        ("\\multicolumn{3}{l}{\\textit{Age distribution}} \\\\", None),
        (f"Age 18--30           & {ro['age_young']}\\%  & {age['18-30']:.1f}\\% \\\\", None),
        (f"Age 30--50           & {ro['age_mid']}\\% & {age['31-50']:.1f}\\% \\\\", None),
        (f"Age 51+              & {ro['age_old']}\\% & {age['51+']:.1f}\\% \\\\", None),
        ("\\midrule", None),
        ("\\multicolumn{3}{l}{\\textit{Interaction type}} \\\\", None),
        (f"Monologue (1 speaker)         & {ro['monologue']}\\% & {inter['Monologue']:.1f}\\% \\\\", None),
        (f"Interview/Debate (2 speakers) & {ro['interview']}\\% & {inter['Interview']:.1f}\\% \\\\", None),
        (f"Other (3+ speakers)           & {ro['other_int']}\\% & {inter['Other']:.1f}\\% \\\\", None),
    ]
    for line, _ in rows:
        print(line)


# RO reference values (from the existing paper table)
RO_REFERENCE = {
    "studio_ro": "68.8", "indoor_ro": "27.9", "outdoor_ro": "2.9",
    "noise_low": "88.2", "noise_med": "11.2", "noise_high": "0.6",
    "top_share": "40.6", "other_share": "59.4",
    "male": "54.7", "female": "45.3",
    "age_young": "1.8", "age_mid": "64.7", "age_old": "32.9",
    "monologue": "27.1", "interview": "60.3", "other_int": "12.6",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MD corpus metadata stats.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR_DEFAULT)
    parser.add_argument("--duration-weighted", action="store_true",
                        help="Use duration-weighted gender/age (default: speaker-count).")
    args = parser.parse_args()

    videos = read_csv(args.data_dir / "videos_master.csv")
    speakers = read_csv(args.data_dir / "speakers_registry.csv")
    segments = read_csv(args.data_dir / "segments_index.csv")
    proposals = read_csv(args.data_dir / "auto_tag_workdir" / "proposals.csv")

    print(f"Loaded: {len(videos)} videos, {len(speakers)} speakers, "
          f"{len(segments):,} segments, {len(proposals)} auto-tag proposals")

    env = env_stats(proposals)
    noise = noise_stats(proposals)
    src = source_stats(videos)
    gender = gender_stats(speakers, args.duration_weighted)
    age = age_stats(speakers, args.duration_weighted)
    inter = interaction_stats(segments)

    print_section("Recording environment", [(k, v) for k, v in env.items() if not k.startswith("_")],
                  note=env["_note"])
    print_section("Audio conditions (background noise)",
                  [(k, v) for k, v in noise.items() if not k.startswith("_")], note=noise["_note"])
    print_section(f"Source diversity (top = '{src['_top_name']}')",
                  [(k, v) for k, v in src.items() if not k.startswith("_")])
    print_section(f"Gender distribution ({gender['_weighting']}-weighted)",
                  [(k, v) for k, v in gender.items() if not k.startswith("_")])
    print_section(f"Age distribution ({age['_weighting']}-weighted)",
                  [(k, v) for k, v in age.items() if not k.startswith("_")])
    print_section("Interaction type (distinct speaker_ids per video)",
                  [(k, v) for k, v in inter.items() if not k.startswith("_")])

    print_latex(env, noise, src, gender, age, inter, RO_REFERENCE)


if __name__ == "__main__":
    main()

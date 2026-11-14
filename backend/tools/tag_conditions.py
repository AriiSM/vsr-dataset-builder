"""
Interactive CLI to tag `environment` + `background_noise` per video by
opening each YouTube URL in the browser and reading a single keystroke.

Run:
    python backend/tools/tag_conditions.py
    python backend/tools/tag_conditions.py --all          # re-tag everything, not just untagged
    python backend/tools/tag_conditions.py --channel "Privesc.Eu Moldova"
    python backend/tools/tag_conditions.py --start md_050 # resume from a given id

The script saves after every video, so Ctrl+C is safe.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "catalog" / "videos_master.csv"

ENV_CHOICES = {
    "1": "studio",
    "2": "indoor",
    "3": "outdoor",
    "4": "mixed",
    "5": "unknown",
}
NOISE_CHOICES = {
    "1": "none",
    "2": "low",
    "3": "moderate",
    "4": "high",
}


def _backup(csv: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = csv.with_suffix(f".csv.bak.{ts}.tag-conditions")
    shutil.copy2(csv, bak)
    return bak


def _prompt_choice(label: str, choices: dict[str, str], extras: dict[str, str]) -> str | None:
    """Read one keystroke. Returns the chosen value, or one of the extras
    (e.g. 'skip', 'back', 'quit'), or None on empty input."""
    opts = "  ".join(f"[{k}]{v}" for k, v in choices.items())
    extra_opts = "  ".join(f"[{k}]{v}" for k, v in extras.items())
    while True:
        raw = input(f"  {label}  {opts}    {extra_opts}\n  > ").strip().lower()
        if not raw:
            continue
        if raw in choices:
            return choices[raw]
        if raw in extras:
            return extras[raw]
        print(f"  ! invalid: '{raw}' — try again")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true",
                   help="Iterate all videos (re-tag already-tagged ones too).")
    p.add_argument("--channel", default=None,
                   help="Filter to a specific source_channel.")
    p.add_argument("--start", default=None,
                   help="Start from this video_id (skip everything before).")
    p.add_argument("--no-browser", action="store_true",
                   help="Don't open URLs in browser — just print them.")
    args = p.parse_args()

    if not CSV_PATH.exists():
        print(f"ERROR: not found: {CSV_PATH}", file=sys.stderr)
        return 1

    df = pd.read_csv(CSV_PATH)
    bak = _backup(CSV_PATH)
    print(f"Backup: {bak.name}\n")

    # Build the work queue
    mask = pd.Series(True, index=df.index)
    if not args.all:
        mask &= df["environment"].isna() | df["background_noise"].isna()
    if args.channel:
        mask &= df["source_channel"].fillna("") == args.channel
    queue_ids = df.loc[mask, "video_id"].tolist()

    if args.start:
        if args.start in queue_ids:
            queue_ids = queue_ids[queue_ids.index(args.start):]
        else:
            print(f"WARNING: --start '{args.start}' not in queue; ignored.\n")

    if not queue_ids:
        print("Nothing to tag. Use --all to re-tag completed entries.")
        return 0

    print(f"Queue: {len(queue_ids)} videos\n"
          f"Keys: env [1-5], noise [1-4], s=skip, b=back, q=quit\n"
          f"      'k' = keep current value during noise step\n")

    history: list[str] = []  # video_ids already processed (for 'back')
    idx = 0
    quit_now = False

    while idx < len(queue_ids) and not quit_now:
        vid = queue_ids[idx]
        row = df[df["video_id"] == vid].iloc[0]
        url = str(row.get("youtube_url", "") or "")
        title = str(row.get("title", "") or "")
        channel = str(row.get("source_channel", "") or "—")
        cur_env = row.get("environment")
        cur_noise = row.get("background_noise")

        print("─" * 78)
        print(f"[{idx + 1}/{len(queue_ids)}] {vid}    channel: {channel}")
        print(f"  title:  {title[:120]}")
        print(f"  url:    {url}")
        print(f"  current: env={cur_env!r}  noise={cur_noise!r}")

        if url and not args.no_browser:
            try:
                webbrowser.open(url, new=0)
            except Exception as e:
                print(f"  (could not open browser: {e})")

        # Step 1: environment
        env = _prompt_choice(
            "ENVIRONMENT?",
            ENV_CHOICES,
            {"s": "__skip__", "b": "__back__", "q": "__quit__"},
        )
        if env == "__skip__":
            idx += 1
            continue
        if env == "__quit__":
            quit_now = True
            break
        if env == "__back__":
            if history:
                # Jump back to last processed; insert it back at the queue front-ish
                prev = history.pop()
                # Re-insert prev right at idx (so next iter processes prev)
                queue_ids.insert(idx, prev)
                continue
            else:
                print("  ! no previous video to go back to")
                continue

        # Step 2: noise
        noise_extras = {"s": "__skip__", "k": "__keep__", "q": "__quit__"}
        noise = _prompt_choice(
            "NOISE?    ",
            NOISE_CHOICES,
            noise_extras,
        )
        if noise == "__skip__":
            idx += 1
            continue
        if noise == "__quit__":
            quit_now = True
            break
        if noise == "__keep__":
            noise = cur_noise if pd.notna(cur_noise) else None

        # Save
        df.loc[df["video_id"] == vid, "environment"] = env
        if noise is not None:
            df.loc[df["video_id"] == vid, "background_noise"] = noise
        df.to_csv(CSV_PATH, index=False)
        history.append(vid)
        print(f"  ✓ saved: env={env}  noise={noise}")
        idx += 1

    # Final summary
    print("\n" + "═" * 78)
    df2 = pd.read_csv(CSV_PATH)
    print(f"Done. Total videos: {len(df2)}")
    print(f"  env  tagged: {int(df2['environment'].notna().sum())} / {len(df2)}")
    print(f"  noise tagged: {int(df2['background_noise'].notna().sum())} / {len(df2)}")
    print(f"\nBackup kept: {bak.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

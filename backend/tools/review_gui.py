#!/usr/bin/env python3
"""
Annotation Review GUI - Romanian VSR Dataset

Lets you play each processed segment and:
  - Accept as-is
  - Modify the transcription (full text + word-level timings)
  - Drop the segment (deletes video + annotation, updates index + Excel)

Works with both pipeline v1 and v2 output directories.

Usage:
    python backend/tools/review_gui.py
    python backend/tools/review_gui.py --data ./data
    python backend/tools/review_gui.py --data /path/to/other/project/data
"""

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import pandas as pd

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Pillow is required: pip install Pillow")
    sys.exit(1)

import threading

# Audio backend: winsound (Windows built-in, no extra install)
try:
    import winsound
    _AUDIO_BACKEND = "winsound"
except ImportError:
    _AUDIO_BACKEND = None
    print("Note: winsound not available - audio playback disabled")

# DATA HELPERS

def catalog_dir(data_dir: Path) -> Path:
    """Storage v2 keeps metadata under data/catalog; fall back to the legacy
    data/metadata layout for datasets produced before Faza 6.5."""
    v2 = Path(data_dir) / "catalog"
    if (v2 / "segments_index.csv").exists() or v2.exists():
        return v2
    return Path(data_dir) / "metadata"


def parse_annotation(path: Path) -> dict:
    """Parse an LRS2-format annotation file."""
    text = path.read_text(encoding="utf-8").strip()
    lines = text.splitlines()

    result = {"text": "", "conf": 2, "words": []}
    word_section = False

    for line in lines:
        if line.startswith("Text:"):
            result["text"] = line[len("Text:"):].strip()
        elif line.startswith("Conf:"):
            try:
                result["conf"] = int(line[len("Conf:"):].strip())
            except ValueError:
                pass
        elif line.startswith("WORD START END"):
            word_section = True
        elif word_section and line.strip():
            parts = line.split()
            if len(parts) >= 4:
                result["words"].append({
                    "word": parts[0],
                    "start": float(parts[1]),
                    "end": float(parts[2]),
                    "score": float(parts[3]),
                })
            elif len(parts) == 3:
                result["words"].append({
                    "word": parts[0],
                    "start": float(parts[1]),
                    "end": float(parts[2]),
                    "score": 0.0,
                })

    return result

def write_annotation(path: Path, text: str, conf: int, words: List[dict]):
    """Write an LRS2-format annotation file."""
    lines = [
        f"Text:  {text}",
        f"Conf:  {conf}",
        "WORD START END ASDSCORE",
    ]
    for w in words:
        lines.append(f"{w['word']} {w['start']:.2f} {w['end']:.2f} {w['score']:.1f}")
    path.write_text("\n".join(lines), encoding="utf-8")

def collect_segments(data_dir: Path, video_filter: Optional[str] = None) -> List[dict]:
    """
    Scan processed/ and annotations/ to build the segment list.
    Falls back to segments_index.csv if available, then scans filesystem.
    """
    processed_dir = data_dir / "processed"
    annotations_dir = data_dir / "annotations"   # legacy fallback (storage v1)
    index_path = catalog_dir(data_dir) / "segments_index.csv"

    segments = []

    if index_path.exists():
        df = pd.read_csv(index_path)
        df = df.dropna(subset=["segment_id", "video_id"])

    if index_path.exists() and len(df) > 0:
        for _, row in df.iterrows():
            seg_id = str(row["segment_id"])
            video_id = str(row["video_id"])
            video_path = processed_dir / video_id / f"{seg_id}.mp4"
            anno_path = processed_dir / video_id / "text" / f"{seg_id}.txt"
            if not anno_path.exists():
                anno_path = annotations_dir / video_id / f"{seg_id}.txt"

            if video_path.exists() and anno_path.exists():
                segments.append({
                    "segment_id": seg_id,
                    "video_id": video_id,
                    "video_path": video_path,
                    "anno_path": anno_path,
                    "duration": float(row.get("duration", 0)),
                    "text": str(row.get("text", "")),
                    "status": "pending",
                })
    else:
        # Fallback: scan filesystem
        if processed_dir.exists():
            for video_dir in sorted(processed_dir.iterdir()):
                if not video_dir.is_dir():
                    continue
                video_id = video_dir.name
                for video_file in sorted(video_dir.glob("*.mp4")):
                    seg_id = video_file.stem
                    anno_path = processed_dir / video_id / "text" / f"{seg_id}.txt"
                    if not anno_path.exists():
                        anno_path = annotations_dir / video_id / f"{seg_id}.txt"
                    if anno_path.exists():
                        cap = cv2.VideoCapture(str(video_file))
                        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        cap.release()
                        duration = frames / fps if fps > 0 else 0.0
                        ann = parse_annotation(anno_path)
                        segments.append({
                            "segment_id": seg_id,
                            "video_id": video_id,
                            "video_path": video_file,
                            "anno_path": anno_path,
                            "duration": duration,
                            "text": ann["text"],
                            "status": "pending",
                        })

    if video_filter:
        segments = [s for s in segments if s["video_id"] == video_filter]

    return segments

def drop_segment_files(seg: dict, data_dir: Path):
    """Delete video and annotation files for a segment."""
    seg["video_path"].unlink(missing_ok=True)
    seg["anno_path"].unlink(missing_ok=True)

    # Remove empty parent directories
    for path in [seg["video_path"].parent, seg["anno_path"].parent]:
        try:
            if path.exists() and not any(path.iterdir()):
                path.rmdir()
        except Exception:
            pass

def update_index(data_dir: Path, removed_ids: set):
    """Remove dropped segments from segments_index.csv."""
    index_path = catalog_dir(data_dir) / "segments_index.csv"
    if not index_path.exists() or not removed_ids:
        return
    df = pd.read_csv(index_path)
    df = df[~df["segment_id"].astype(str).isin(removed_ids)]
    df.to_csv(index_path, index=False)

def set_excel_video_status(data_dir: Path, video_id: str, status: str):
    """Write a new status value for one video in videos_master.csv."""
    csv_path = catalog_dir(data_dir) / "videos_master.csv"
    if not csv_path.exists():
        return
    try:
        df = pd.read_csv(csv_path)
        mask = df["video_id"].astype(str) == video_id
        if not mask.any():
            return
        df = df.astype(object)
        df.loc[mask, "status"] = status
        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Warning: could not update CSV status for {video_id}: {e}")

def update_excel(data_dir: Path, affected_video_ids: set, remaining_segments: List[dict]):
    """Update videos_master.csv totals for affected videos."""
    csv_path = catalog_dir(data_dir) / "videos_master.csv"
    if not csv_path.exists():
        return
    try:
        df = pd.read_csv(csv_path)
        for col in df.select_dtypes(include=["float64", "object"]).columns:
            df[col] = df[col].astype(object)

        for video_id in affected_video_ids:
            segs = [s for s in remaining_segments if s["video_id"] == video_id]
            mask = df["video_id"].astype(str) == video_id
            if not mask.any():
                continue
            idx = df.index[mask][0]
            df.loc[idx, "total_segments"] = len(segs)
            df.loc[idx, "total_duration_extracted"] = sum(s["duration"] for s in segs)

        df.to_csv(csv_path, index=False)
    except Exception as e:
        print(f"Warning: could not update CSV: {e}")

# EDIT DIALOG

class EditDialog(tk.Toplevel):
    """
    Dialog for editing the transcription of one segment.

    Shows:
      - Editable text field (full sentence)
      - Word-by-word table (double-click to edit a cell)

    Returns the updated annotation dict via self.result.
    """

    def __init__(self, parent, annotation: dict):
        super().__init__(parent)
        self.title("Edit Transcription")
        self.resizable(True, True)
        self.result = None
        self._ann = {
            "text": annotation["text"],
            "conf": annotation["conf"],
            "words": [w.copy() for w in annotation["words"]],
        }

        self._build_ui()
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.geometry("620x520")
        self.minsize(500, 400)

    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        # Full text
        tk.Label(self, text="Full text:", anchor="w", font=("", 10, "bold")).pack(
            fill="x", **pad
        )
        text_frame = tk.Frame(self)
        text_frame.pack(fill="x", padx=8, pady=(0, 6))
        text_frame.columnconfigure(0, weight=1)
        self._text_widget = tk.Text(
            text_frame, font=("Consolas", 11),
            wrap="word", height=3,
            relief="sunken", borderwidth=1,
        )
        self._text_widget.insert("1.0", self._ann["text"])
        self._text_widget.grid(row=0, column=0, sticky="ew")
        text_sb = ttk.Scrollbar(text_frame, orient="vertical", command=self._text_widget.yview)
        self._text_widget.configure(yscrollcommand=text_sb.set)
        text_sb.grid(row=0, column=1, sticky="ns")

        # Word table
        tk.Label(self, text="Word-level timings (double-click to edit):",
                 anchor="w", font=("", 10, "bold")).pack(fill="x", **pad)

        frame = tk.Frame(self)
        frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        cols = ("word", "start", "end", "score")
        self._tree = ttk.Treeview(frame, columns=cols, show="headings", height=10)
        for col, w in [("word", 130), ("start", 80), ("end", 80), ("score", 80)]:
            self._tree.heading(col, text=col.upper())
            self._tree.column(col, width=w, anchor="center" if col != "word" else "w")

        sb = ttk.Scrollbar(frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        self._tree.bind("<Double-1>", self._on_cell_double_click)
        self._populate_tree()

        # Buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=8)
        tk.Button(btn_frame, text="Save", width=12, bg="#4CAF50", fg="white",
                  command=self._on_save).pack(side="right", padx=4)
        tk.Button(btn_frame, text="Cancel", width=12,
                  command=self._on_cancel).pack(side="right", padx=4)
        tk.Button(btn_frame, text="Sync text → words",
                  command=self._sync_text_to_words).pack(side="left", padx=4)

    def _populate_tree(self):
        for item in self._tree.get_children():
            self._tree.delete(item)
        for w in self._ann["words"]:
            self._tree.insert("", "end", values=(
                w["word"], f"{w['start']:.2f}", f"{w['end']:.2f}", f"{w['score']:.1f}"
            ))

    def _on_cell_double_click(self, event):
        """Open an inline editor for the clicked cell."""
        region = self._tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        row_id = self._tree.identify_row(event.y)
        col_id = self._tree.identify_column(event.x)
        col_idx = int(col_id.lstrip("#")) - 1
        col_names = ("word", "start", "end", "score")

        x, y, w, h = self._tree.bbox(row_id, col_id)
        current = self._tree.item(row_id, "values")[col_idx]

        entry_var = tk.StringVar(value=current)
        entry = tk.Entry(self._tree, textvariable=entry_var, font=("Consolas", 10))
        entry.place(x=x, y=y, width=w, height=h)
        entry.focus_set()
        entry.select_range(0, "end")

        def _commit(event=None):
            new_val = entry_var.get().strip()
            vals = list(self._tree.item(row_id, "values"))
            vals[col_idx] = new_val
            self._tree.item(row_id, values=vals)
            entry.destroy()

        entry.bind("<Return>", _commit)
        entry.bind("<Tab>", _commit)
        entry.bind("<FocusOut>", _commit)
        entry.bind("<Escape>", lambda e: entry.destroy())

    def _sync_text_to_words(self):
        """
        Split the text field into tokens and try to update word labels
        in the table. If counts differ, redistribute timings evenly.
        """
        new_words = self._text_widget.get("1.0", "end").strip().split()
        old_vals = [self._tree.item(r, "values") for r in self._tree.get_children()]

        if len(new_words) == len(old_vals):
            # Same count - just update word column
            for i, row_id in enumerate(self._tree.get_children()):
                vals = list(self._tree.item(row_id, "values"))
                vals[0] = new_words[i]
                self._tree.item(row_id, values=vals)
        else:
            # Different count - redistribute timings evenly
            if old_vals:
                t_start = float(old_vals[0][1])
                t_end = float(old_vals[-1][2])
            else:
                t_start, t_end = 0.0, 1.0

            step = (t_end - t_start) / max(len(new_words), 1)
            for item in self._tree.get_children():
                self._tree.delete(item)
            for i, word in enumerate(new_words):
                ws = round(t_start + i * step, 2)
                we = round(t_start + (i + 1) * step, 2)
                self._tree.insert("", "end", values=(word, f"{ws:.2f}", f"{we:.2f}", "0.0"))

    def _on_save(self):
        self._sync_text_to_words()
        words = []
        for row_id in self._tree.get_children():
            vals = self._tree.item(row_id, "values")
            try:
                words.append({
                    "word": str(vals[0]),
                    "start": float(vals[1]),
                    "end": float(vals[2]),
                    "score": float(vals[3]),
                })
            except (ValueError, IndexError):
                pass

        self.result = {
            "text": self._text_widget.get("1.0", "end").strip(),
            "conf": self._ann["conf"],
            "words": words,
        }
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()

# TRIM DIALOG

class TrimDialog(tk.Toplevel):
    """
    Dialog for trimming the start and/or end of a video segment.

    Two sliders set the trim-in and trim-out points (in seconds).
    Moving a slider previews that frame on the canvas.
    The Play button plays exactly the trimmed range so you can verify the cut.

    Returns via self.result = (trim_start_s, trim_end_s), or None if cancelled.
    """

    PREVIEW_W = 480
    PREVIEW_H = 320

    def __init__(self, parent, video_path: Path, annotation: dict):
        super().__init__(parent)
        self.title("Trim Video")
        self.resizable(True, True)
        self.result = None

        self._video_path = video_path
        self._ann = annotation
        self._cap = cv2.VideoCapture(str(video_path))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = self._total_frames / self._fps

        self._trim_start = tk.DoubleVar(value=0.0)
        self._trim_end = tk.DoubleVar(value=self._duration)

        # Playback state
        self._playing = False
        self._after_id = None
        self._play_start_time: float = 0.0
        self._play_start_frame: int = 0
        self._audio_file: Optional[Path] = None

        self._build_ui()
        self._show_frame_at(0.0)
        self._update_info()

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.geometry("600x640")
        self.minsize(500, 540)

    def _build_ui(self):
        # Video preview
        canvas_frame = tk.Frame(self, bg="#1a1a1a")
        canvas_frame.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        self._canvas = tk.Canvas(
            canvas_frame,
            width=self.PREVIEW_W, height=self.PREVIEW_H,
            bg="#1a1a1a", highlightthickness=0,
        )
        self._canvas.pack(expand=True)

        self._preview_label = tk.Label(
            self, text="", font=("Consolas", 9), fg="#777", bg=self["bg"]
        )
        self._preview_label.pack(pady=(2, 0))

        # Play button
        play_row = tk.Frame(self)
        play_row.pack(pady=(2, 0))
        self._play_btn = tk.Button(
            play_row, text="▶  Play trimmed range",
            bg="#0078d7", fg="white", relief="flat",
            padx=12, pady=4,
            command=self._toggle_play,
        )
        self._play_btn.pack()

        # Sliders
        sliders = tk.Frame(self)
        sliders.pack(fill="x", padx=12, pady=6)
        sliders.columnconfigure(1, weight=1)

        tk.Label(sliders, text="Trim start:", anchor="w", width=10).grid(
            row=0, column=0, sticky="w", pady=3)
        self._start_slider = ttk.Scale(
            sliders, from_=0.0, to=self._duration,
            orient="horizontal", variable=self._trim_start,
            command=self._on_start_changed,
        )
        self._start_slider.grid(row=0, column=1, sticky="ew", padx=6)
        self._start_val_lbl = tk.Label(sliders, text="0.00s", width=8, anchor="w")
        self._start_val_lbl.grid(row=0, column=2, sticky="w")

        tk.Label(sliders, text="Trim end:", anchor="w", width=10).grid(
            row=1, column=0, sticky="w", pady=3)
        self._end_slider = ttk.Scale(
            sliders, from_=0.0, to=self._duration,
            orient="horizontal", variable=self._trim_end,
            command=self._on_end_changed,
        )
        self._end_slider.grid(row=1, column=1, sticky="ew", padx=6)
        self._end_val_lbl = tk.Label(
            sliders, text=f"{self._duration:.2f}s", width=8, anchor="w"
        )
        self._end_val_lbl.grid(row=1, column=2, sticky="w")

        # Info line
        self._info_label = tk.Label(self, text="", font=("", 9), fg="#555")
        self._info_label.pack(pady=(0, 4))

        # Apply / Cancel buttons
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=8)

        tk.Button(btn_frame, text="Preview start",
                  command=lambda: self._show_frame_at(self._trim_start.get())
                  ).pack(side="left", padx=4)
        tk.Button(btn_frame, text="Preview end",
                  command=lambda: self._show_frame_at(self._trim_end.get())
                  ).pack(side="left", padx=4)

        tk.Button(btn_frame, text="Apply Trim", width=14,
                  bg="#27ae60", fg="white",
                  command=self._on_apply).pack(side="right", padx=4)
        tk.Button(btn_frame, text="Cancel", width=10,
                  command=self._on_cancel).pack(side="right", padx=4)

    # Slider callbacks
    def _on_start_changed(self, val):
        self._stop_play()
        s = float(val)
        end = self._trim_end.get()
        if s >= end - 0.04:
            s = max(0.0, end - 0.04)
            self._trim_start.set(s)
        self._start_val_lbl.config(text=f"{s:.2f}s")
        self._show_frame_at(s)
        self._update_info()

    def _on_end_changed(self, val):
        self._stop_play()
        e = float(val)
        start = self._trim_start.get()
        if e <= start + 0.04:
            e = min(self._duration, start + 0.04)
            self._trim_end.set(e)
        self._end_val_lbl.config(text=f"{e:.2f}s")
        self._show_frame_at(e)
        self._update_info()

    # Playback
    def _extract_trimmed_audio(self, start_s: float, end_s: float) -> Optional[Path]:
        """Extract [start_s, end_s] audio from the video to a temp WAV."""
        if _AUDIO_BACKEND is None:
            return None
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = "ffmpeg"
        tmp = Path(tempfile.gettempdir()) / f"_trim_preview_{self._video_path.stem}.wav"
        try:
            result = subprocess.run(
                [ffmpeg_exe, "-y",
                 "-ss", f"{start_s:.4f}",
                 "-i", str(self._video_path),
                 "-t", f"{end_s - start_s:.4f}",
                 "-vn", "-acodec", "pcm_s16le",
                 str(tmp)],
                capture_output=True,
            )
            if result.returncode != 0:
                return None
            return tmp
        except Exception:
            return None

    def _play_audio(self):
        if _AUDIO_BACKEND != "winsound" or self._audio_file is None:
            return
        wav = str(self._audio_file)
        threading.Thread(
            target=winsound.PlaySound,
            args=(wav, winsound.SND_FILENAME | winsound.SND_ASYNC),
            daemon=True,
        ).start()

    def _stop_audio(self):
        if _AUDIO_BACKEND == "winsound":
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    def _toggle_play(self):
        if self._playing:
            self._stop_play()
        else:
            start_s = self._trim_start.get()
            self._audio_file = self._extract_trimmed_audio(start_s, self._trim_end.get())
            self._playing = True
            self._play_btn.config(text="⏸  Pause")
            self._play_start_frame = int(start_s * self._fps)
            self._play_start_time = time.perf_counter()
            self._play_audio()
            self._play_loop()

    def _play_loop(self):
        if not self._playing or self._cap is None:
            return

        end_frame = int(self._trim_end.get() * self._fps)
        elapsed = time.perf_counter() - self._play_start_time
        target_frame = self._play_start_frame + int(elapsed * self._fps)

        if target_frame >= end_frame:
            self._stop_play()
            self._show_frame_at(self._trim_start.get())
            return

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = self._cap.read()
        if ret:
            self._render_frame(frame, target_frame / self._fps)

        self._after_id = self.after(16, self._play_loop)

    def _stop_play(self):
        self._playing = False
        if hasattr(self, "_play_btn"):
            self._play_btn.config(text="▶  Play trimmed range")
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._stop_audio()

    # Frame rendering
    def _render_frame(self, frame, time_s: float):
        """Display a raw BGR frame on the canvas and update the time label."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = min(self.PREVIEW_W / w, self.PREVIEW_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

        img = Image.fromarray(frame_rgb)
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        x_off = (self.PREVIEW_W - new_w) // 2
        y_off = (self.PREVIEW_H - new_h) // 2
        self._canvas.create_image(x_off, y_off, anchor="nw", image=self._photo)
        frame_no = int(time_s * self._fps)
        self._preview_label.config(text=f"t = {time_s:.2f}s  (frame {frame_no})")

    def _show_frame_at(self, time_s: float):
        if self._cap is None:
            return
        frame_no = int(time_s * self._fps)
        frame_no = max(0, min(frame_no, self._total_frames - 1))
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self._cap.read()
        if ret:
            self._render_frame(frame, time_s)

    def _update_info(self):
        start = self._trim_start.get()
        end = self._trim_end.get()
        new_dur = end - start
        words_kept = sum(
            1 for w in self._ann["words"]
            if w["end"] > start and w["start"] < end
        )
        self._info_label.config(
            text=(f"New duration: {new_dur:.2f}s  |  "
                  f"Words kept: {words_kept} / {len(self._ann['words'])}")
        )

    # Apply / Cancel
    def _cleanup_audio(self):
        if self._audio_file and self._audio_file.exists():
            try:
                self._audio_file.unlink()
            except Exception:
                pass
        self._audio_file = None

    def _on_apply(self):
        self._stop_play()
        self._cleanup_audio()
        start = self._trim_start.get()
        end = self._trim_end.get()
        if end - start < 0.04:
            messagebox.showwarning("Invalid trim",
                                   "Trim range is too short.", parent=self)
            return
        if abs(start) < 0.01 and abs(end - self._duration) < 0.01:
            messagebox.showinfo("No change",
                                "Both handles are at their defaults - nothing to trim.",
                                parent=self)
            return
        self.result = (start, end)
        self._cap.release()
        self._cap = None
        self.destroy()

    def _on_cancel(self):
        self._stop_play()
        self._cleanup_audio()
        self.result = None
        if self._cap:
            self._cap.release()
            self._cap = None
        self.destroy()

# MAIN APPLICATION

class ReviewApp:
    """Main review GUI."""

    VIDEO_W = 480
    VIDEO_H = 320

    def __init__(self, root: tk.Tk, data_dir: Path, video_filter: Optional[str] = None):
        self.root = root
        self.data_dir = data_dir
        self._video_filter = video_filter
        title = f"VSR Annotation Review - {data_dir}"
        if video_filter:
            title += f"  [{video_filter}]"
        self.root.title(title)
        self.root.minsize(1100, 640)

        self._segments: List[dict] = []
        self._current_idx: Optional[int] = None

        # Video playback state
        self._cap: Optional[cv2.VideoCapture] = None
        self._playing = False
        self._fps = 25.0
        self._total_frames = 0
        self._after_id = None
        self._play_start_time: float = 0.0
        self._play_start_frame: int = 0

        # Audio playback state
        self._audio_file: Optional[Path] = None

        # Track which videos were affected by drops (for Excel update)
        self._dropped_ids: set = set()
        self._dropped_video_ids: set = set()

        if video_filter and not self._check_video_status(video_filter):
            self.root.after(0, self.root.destroy)
            return

        self._build_ui()
        self._load_segments()

    # UI construction
    def _build_ui(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Main horizontal PanedWindow (allows dragging dividers)
        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.grid(row=0, column=0, sticky="nsew")

        # Left: segment list
        left = tk.Frame(paned, width=220, bg="#f5f5f5")
        left.pack_propagate(False)
        self._build_segment_list(left)
        paned.add(left, weight=0)

        # Center: video + controls
        center = tk.Frame(paned, bg="#1a1a1a")
        self._build_video_panel(center)
        paned.add(center, weight=1)

        # Right: annotation
        right = tk.Frame(paned, width=360, bg="#fafafa")
        right.pack_propagate(False)
        self._build_annotation_panel(right)
        paned.add(right, weight=0)

        # Bottom: action bar
        bottom = tk.Frame(self.root, bg="#2d2d2d", height=56)
        bottom.grid(row=1, column=0, sticky="ew")
        bottom.grid_propagate(False)
        self._build_action_bar(bottom)

    def _build_segment_list(self, parent):
        tk.Label(parent, text="Segments", bg="#f5f5f5",
                 font=("", 11, "bold")).pack(pady=(8, 2))

        # Stats label
        self._stats_label = tk.Label(parent, text="", bg="#f5f5f5",
                                     font=("", 9), fg="#555")
        self._stats_label.pack()

        frame = tk.Frame(parent, bg="#f5f5f5")
        frame.pack(fill="both", expand=True, padx=4, pady=4)

        sb = ttk.Scrollbar(frame, orient="vertical")
        self._listbox = tk.Listbox(
            frame, yscrollcommand=sb.set, selectmode="single",
            font=("Consolas", 9), activestyle="dotbox",
            bg="white", selectbackground="#0078d7", selectforeground="white",
        )
        sb.config(command=self._listbox.yview)
        self._listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._listbox.bind("<<ListboxSelect>>", self._on_list_select)

    def _build_video_panel(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        # Video canvas
        canvas_frame = tk.Frame(parent, bg="#1a1a1a")
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self._canvas = tk.Canvas(
            canvas_frame,
            width=self.VIDEO_W, height=self.VIDEO_H,
            bg="#1a1a1a", highlightthickness=0,
        )
        self._canvas.pack(expand=True)
        self._canvas.create_text(
            self.VIDEO_W // 2, self.VIDEO_H // 2,
            text="Select a segment to preview",
            fill="#888", font=("", 12), tags="placeholder",
        )

        # Seek bar
        ctrl = tk.Frame(parent, bg="#2a2a2a")
        ctrl.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))

        self._seek_var = tk.DoubleVar(value=0.0)
        self._seek_bar = ttk.Scale(
            ctrl, from_=0, to=100, orient="horizontal",
            variable=self._seek_var, command=self._on_seek,
        )
        self._seek_bar.pack(fill="x", padx=8, pady=4)

        self._time_label = tk.Label(
            ctrl, text="0.0s / 0.0s", bg="#2a2a2a", fg="#aaa", font=("Consolas", 9)
        )
        self._time_label.pack()

        # Play/pause button
        btn_row = tk.Frame(parent, bg="#2a2a2a")
        btn_row.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        self._play_btn = tk.Button(
            btn_row, text="▶  Play", width=12,
            bg="#0078d7", fg="white", relief="flat",
            command=self._toggle_play,
        )
        self._play_btn.pack(pady=4)

    def _build_annotation_panel(self, parent):
        tk.Label(parent, text="Annotation", bg="#fafafa",
                 font=("", 11, "bold")).pack(pady=(8, 2))

        # Segment ID
        self._seg_id_label = tk.Label(
            parent, text="-", bg="#fafafa", font=("Consolas", 10), fg="#333"
        )
        self._seg_id_label.pack()

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=8, pady=6)

        # Text
        tk.Label(parent, text="Text:", bg="#fafafa",
                 font=("", 10, "bold"), anchor="w").pack(fill="x", padx=8)

        self._text_display = tk.Text(
            parent, height=1, wrap="word", font=("Consolas", 11),
            bg="#fff", relief="solid", bd=1,
        )
        self._text_display.pack(fill="x", padx=8, pady=(2, 8))
        self._text_display.config(state="disabled")

        # Words
        tk.Label(parent, text="Words:", bg="#fafafa",
                 font=("", 10, "bold"), anchor="w").pack(fill="x", padx=8)

        word_frame = tk.Frame(parent, bg="#fafafa")
        word_frame.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        cols = ("word", "start", "end")
        self._word_tree = ttk.Treeview(word_frame, columns=cols, show="headings")
        for col, w, label in [("word", 130, "WORD"), ("start", 70, "START"), ("end", 70, "END")]:
            self._word_tree.heading(col, text=label)
            self._word_tree.column(col, width=w, anchor="center" if col != "word" else "w")

        wsb = ttk.Scrollbar(word_frame, orient="vertical", command=self._word_tree.yview)
        self._word_tree.configure(yscrollcommand=wsb.set)
        self._word_tree.pack(side="left", fill="both", expand=True)
        wsb.pack(side="right", fill="y")

        # Scores
        self._score_label = tk.Label(
            parent, text="", bg="#fafafa", font=("", 9), fg="#555"
        )
        self._score_label.pack(padx=8, pady=(0, 4))

    def _build_action_bar(self, parent):
        # Status label on the left
        self._action_status = tk.Label(
            parent, text="", bg="#2d2d2d", fg="#aaa", font=("", 10)
        )
        self._action_status.pack(side="left", padx=16)

        # Buttons on the right
        btn_cfg = {"relief": "flat", "width": 16, "pady": 6, "padx": 8}

        tk.Button(
            parent, text="✗  Drop Segment",
            bg="#c0392b", fg="white",
            command=self._action_drop, **btn_cfg
        ).pack(side="right", padx=8, pady=8)

        tk.Button(
            parent, text="✂  Trim Video",
            bg="#8e44ad", fg="white",
            command=self._action_trim, **btn_cfg
        ).pack(side="right", padx=4, pady=8)

        tk.Button(
            parent, text="✎  Modify Text",
            bg="#e67e22", fg="white",
            command=self._action_modify, **btn_cfg
        ).pack(side="right", padx=4, pady=8)

        tk.Button(
            parent, text="✓  Accept",
            bg="#27ae60", fg="white",
            command=self._action_accept, **btn_cfg
        ).pack(side="right", padx=4, pady=8)

        tk.Button(
            parent, text="Save & Quit",
            bg="#555", fg="white",
            command=self._save_and_quit, **btn_cfg
        ).pack(side="right", padx=4, pady=8)

    # Data loading
    def _load_segments(self):
        self._segments = collect_segments(self.data_dir, self._video_filter)
        self._listbox.delete(0, "end")

        for seg in self._segments:
            label = f"[?] {seg['segment_id']}  {seg['duration']:.1f}s"
            self._listbox.insert("end", label)
            self._listbox.itemconfig("end", fg="#333")

        self._update_stats()

        if self._segments:
            self._listbox.select_set(0)
            self._load_segment(0)

    def _update_stats(self):
        total = len(self._segments)
        reviewed = sum(1 for s in self._segments if s["status"] != "pending")
        accepted = sum(1 for s in self._segments if s["status"] == "accepted")
        modified = sum(1 for s in self._segments if s["status"] == "modified")
        dropped_count = len(self._dropped_ids)

        self._stats_label.config(
            text=f"{reviewed}/{total} reviewed  •  "
                 f"{accepted} ✓  {modified} ✎  {dropped_count} ✗"
        )

    def _update_list_item(self, idx: int):
        seg = self._segments[idx]
        status_icons = {
            "pending": "[?]",
            "accepted": "[✓]",
            "modified": "[✎]",
        }
        colors = {
            "pending": "#333",
            "accepted": "#1a7a1a",
            "modified": "#b06000",
        }
        icon = status_icons.get(seg["status"], "[?]")
        label = f"{icon} {seg['segment_id']}  {seg['duration']:.1f}s"
        self._listbox.delete(idx)
        self._listbox.insert(idx, label)
        self._listbox.itemconfig(idx, fg=colors.get(seg["status"], "#333"))
        self._listbox.select_set(idx)

    # Segment loading
    def _on_list_select(self, event):
        sel = self._listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx != self._current_idx:
            self._load_segment(idx)

    def _extract_audio(self, video_path: Path) -> Optional[Path]:
        """Extract audio from video to a temp WAV for playback."""
        if _AUDIO_BACKEND is None:
            return None
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = "ffmpeg"
        tmp = Path(tempfile.gettempdir()) / f"_review_{video_path.stem}.wav"
        try:
            result = subprocess.run(
                [ffmpeg_exe, "-y", "-i", str(video_path),
                 "-vn", "-acodec", "pcm_s16le",
                 str(tmp)],
                capture_output=True,
            )
            if result.returncode != 0:
                print(f"ffmpeg audio extract failed: {result.stderr.decode()[:200]}")
                return None
            print(f"Audio extracted: {tmp} ({tmp.stat().st_size} bytes)")
            return tmp
        except Exception as e:
            print(f"Audio extract exception: {e}")
            return None

    def _play_audio(self):
        """Play self._audio_file in a background thread using winsound."""
        if _AUDIO_BACKEND != "winsound" or self._audio_file is None:
            return
        wav = str(self._audio_file)
        threading.Thread(
            target=winsound.PlaySound,
            args=(wav, winsound.SND_FILENAME | winsound.SND_ASYNC),
            daemon=True,
        ).start()

    def _stop_audio(self):
        """Stop winsound playback."""
        if _AUDIO_BACKEND == "winsound":
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    def _load_segment(self, idx: int):
        self._stop_video()
        self._current_idx = idx
        seg = self._segments[idx]

        # Extract audio for playback
        self._stop_audio()
        if self._audio_file and self._audio_file.exists():
            try:
                self._audio_file.unlink()
            except Exception:
                pass
        self._audio_file = self._extract_audio(seg["video_path"])

        # Load video
        self._cap = cv2.VideoCapture(str(seg["video_path"]))
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._seek_bar.config(to=max(1, self._total_frames - 1))
        self._seek_var.set(0)

        self._show_frame(0)
        self._action_status.config(text=f"Loaded: {seg['segment_id']}")

        # Load annotation
        ann = parse_annotation(seg["anno_path"])
        self._show_annotation(seg, ann)

    def _show_annotation(self, seg: dict, ann: dict):
        self._seg_id_label.config(text=seg["segment_id"])

        self._text_display.config(state="normal")
        self._text_display.delete("1.0", "end")
        self._text_display.insert("1.0", ann["text"])
        # Resize to show all content: measure actual display lines after layout
        self._text_display.update_idletasks()
        display_lines = self._text_display.count("1.0", "end", "displaylines")
        self._text_display.config(height=max(1, display_lines[0] if display_lines else 1))
        self._text_display.config(state="disabled")

        for item in self._word_tree.get_children():
            self._word_tree.delete(item)
        for w in ann["words"]:
            self._word_tree.insert("", "end", values=(w["word"], f"{w['start']:.2f}", f"{w['end']:.2f}"))

        dur = seg["duration"]
        self._score_label.config(
            text=f"Duration: {dur:.2f}s  |  Words: {len(ann['words'])}  |  Conf: {ann['conf']}"
        )

    # Video playback
    def _show_frame(self, frame_no: int):
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self._cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = min(self.VIDEO_W / w, self.VIDEO_H / h)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))

        img = Image.fromarray(frame_rgb)
        self._photo = ImageTk.PhotoImage(img)

        self._canvas.delete("all")
        x_off = (self.VIDEO_W - new_w) // 2
        y_off = (self.VIDEO_H - new_h) // 2
        self._canvas.create_image(x_off, y_off, anchor="nw", image=self._photo)

        pos_s = frame_no / self._fps
        total_s = self._total_frames / self._fps
        self._time_label.config(text=f"{pos_s:.1f}s / {total_s:.1f}s")
        self._seek_var.set(frame_no)

    def _toggle_play(self):
        if self._playing:
            self._stop_video()
        else:
            self._playing = True
            self._play_btn.config(text="⏸  Pause")
            # Reset to frame 0 for audio/video sync
            if self._cap:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._play_start_frame = 0
            self._play_start_time = time.perf_counter()
            self._play_audio()
            self._play_loop()

    def _play_loop(self):
        if not self._playing or self._cap is None:
            return

        # Time-based frame selection keeps video in sync with audio
        elapsed = time.perf_counter() - self._play_start_time
        target_frame = self._play_start_frame + int(elapsed * self._fps)

        if target_frame >= self._total_frames:
            self._stop_video()
            self._show_frame(0)
            return

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = self._cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            scale = min(self.VIDEO_W / w, self.VIDEO_H / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
            img = Image.fromarray(frame_rgb)
            self._photo = ImageTk.PhotoImage(img)
            self._canvas.delete("all")
            x_off = (self.VIDEO_W - new_w) // 2
            y_off = (self.VIDEO_H - new_h) // 2
            self._canvas.create_image(x_off, y_off, anchor="nw", image=self._photo)

            pos_s = target_frame / self._fps
            total_s = self._total_frames / self._fps
            self._time_label.config(text=f"{pos_s:.1f}s / {total_s:.1f}s")
            self._seek_var.set(target_frame)

        self._after_id = self.root.after(16, self._play_loop)  # ~60 Hz UI refresh

    def _stop_video(self):
        self._playing = False
        self._play_btn.config(text="▶  Play")
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self._stop_audio()

    def _on_seek(self, val):
        if self._cap is None:
            return
        frame_no = int(float(val))
        if not self._playing:
            self._show_frame(frame_no)

    # Actions
    def _action_accept(self):
        if self._current_idx is None:
            return
        self._segments[self._current_idx]["status"] = "accepted"
        self._update_list_item(self._current_idx)
        self._update_stats()
        self._action_status.config(text="Accepted.")
        self._advance_to_next()

    def _action_modify(self):
        if self._current_idx is None:
            return
        seg = self._segments[self._current_idx]
        ann = parse_annotation(seg["anno_path"])

        dialog = EditDialog(self.root, ann)
        self.root.wait_window(dialog)

        if dialog.result is None:
            return  # Cancelled

        new_ann = dialog.result

        # Update the annotation file
        write_annotation(seg["anno_path"], new_ann["text"], new_ann["conf"], new_ann["words"])

        # Refresh display
        self._show_annotation(seg, new_ann)

        # Update segments_index.csv text column
        index_path = self.catalog_dir(data_dir) / "segments_index.csv"
        if index_path.exists():
            try:
                df = pd.read_csv(index_path)
                mask = df["segment_id"].astype(str) == seg["segment_id"]
                if mask.any():
                    df.loc[mask, "text"] = new_ann["text"]
                    df.loc[mask, "num_words"] = len(new_ann["words"])
                    df.to_csv(index_path, index=False)
            except Exception as e:
                print(f"Warning: could not update index: {e}")

        self._segments[self._current_idx]["text"] = new_ann["text"]
        self._segments[self._current_idx]["status"] = "modified"
        self._update_list_item(self._current_idx)
        self._update_stats()
        self._action_status.config(text="Saved modifications.")
        self._advance_to_next()

    def _action_trim(self):
        if self._current_idx is None:
            return
        seg = self._segments[self._current_idx]
        ann = parse_annotation(seg["anno_path"])

        # Release main cap so TrimDialog can open its own without contention
        self._stop_video()
        if self._cap:
            self._cap.release()
            self._cap = None

        dialog = TrimDialog(self.root, seg["video_path"], ann)
        self.root.wait_window(dialog)

        if dialog.result is None:
            self._load_segment(self._current_idx)
            return

        trim_start_s, trim_end_s = dialog.result
        duration_s = trim_end_s - trim_start_s

        # Find ffmpeg
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg_exe = "ffmpeg"

        # Trim video file
        video_path = seg["video_path"]
        tmp_path = video_path.with_suffix(".trimtmp.mp4")

        try:
            result = subprocess.run(
                [
                    ffmpeg_exe, "-y",
                    "-ss", f"{trim_start_s:.4f}",
                    "-i", str(video_path),
                    "-t", f"{duration_s:.4f}",
                    "-c:v", "libx264", "-c:a", "aac",
                    str(tmp_path),
                ],
                capture_output=True,
            )
            if result.returncode != 0:
                messagebox.showerror(
                    "Trim failed",
                    f"ffmpeg error:\n{result.stderr.decode()[:400]}",
                    parent=self.root,
                )
                if tmp_path.exists():
                    tmp_path.unlink()
                self._load_segment(self._current_idx)
                return

            video_path.unlink()
            tmp_path.rename(video_path)

        except Exception as e:
            messagebox.showerror("Trim failed", str(e), parent=self.root)
            if tmp_path.exists():
                tmp_path.unlink()
            self._load_segment(self._current_idx)
            return

        # Adjust word timestamps
        # Keep words that overlap the kept range; shift by trim_start_s.
        new_words = []
        for w in ann["words"]:
            if w["end"] <= trim_start_s or w["start"] >= trim_end_s:
                continue  # entirely outside kept range - drop
            new_w = w.copy()
            new_w["start"] = round(max(0.0, w["start"] - trim_start_s), 2)
            new_w["end"] = round(min(duration_s, w["end"] - trim_start_s), 2)
            new_words.append(new_w)

        new_text = " ".join(w["word"] for w in new_words)
        write_annotation(seg["anno_path"], new_text, ann["conf"], new_words)

        # Update in-memory segment & CSV index
        seg["duration"] = duration_s
        index_path = self.catalog_dir(data_dir) / "segments_index.csv"
        if index_path.exists():
            try:
                df = pd.read_csv(index_path)
                mask = df["segment_id"].astype(str) == seg["segment_id"]
                if mask.any():
                    df.loc[mask, "duration"] = round(duration_s, 3)
                    df.loc[mask, "num_words"] = len(new_words)
                    df.loc[mask, "text"] = new_text
                    df.to_csv(index_path, index=False)
            except Exception as e:
                print(f"Warning: could not update index: {e}")

        seg["status"] = "modified"
        self._update_list_item(self._current_idx)
        self._update_stats()
        self._action_status.config(
            text=f"Trimmed {trim_start_s:.2f}s – {trim_end_s:.2f}s  →  {duration_s:.2f}s"
        )
        self._load_segment(self._current_idx)

    def _action_drop(self):
        if self._current_idx is None:
            return
        seg = self._segments[self._current_idx]

        if not messagebox.askyesno(
            "Drop Segment",
            f"Permanently delete '{seg['segment_id']}'?\n\n"
            f"Text: {seg['text'][:80]}\n\n"
            "This cannot be undone.",
            icon="warning",
        ):
            return

        self._stop_video()
        if self._cap:
            self._cap.release()
            self._cap = None

        # Delete files
        drop_segment_files(seg, self.data_dir)

        # Track for index/Excel update
        self._dropped_ids.add(seg["segment_id"])
        self._dropped_video_ids.add(seg["video_id"])

        # Remove from in-memory list
        removed_idx = self._current_idx
        self._segments.pop(removed_idx)
        self._listbox.delete(removed_idx)

        # Update index now (immediate)
        update_index(self.data_dir, {seg["segment_id"]})

        self._update_stats()
        self._action_status.config(text=f"Dropped: {seg['segment_id']}")

        # Load next (or previous) segment
        if not self._segments:
            self._current_idx = None
            self._canvas.delete("all")
            self._canvas.create_text(
                self.VIDEO_W // 2, self.VIDEO_H // 2,
                text="No segments remaining",
                fill="#888", font=("", 12),
            )
            return

        next_idx = min(removed_idx, len(self._segments) - 1)
        self._current_idx = None
        self._listbox.select_clear(0, "end")
        self._listbox.select_set(next_idx)
        self._load_segment(next_idx)

    def _advance_to_next(self):
        """Move to next pending segment, or wrap around."""
        if not self._segments or self._current_idx is None:
            return

        # Find next pending segment
        n = len(self._segments)
        for offset in range(1, n + 1):
            idx = (self._current_idx + offset) % n
            if self._segments[idx]["status"] == "pending":
                self._listbox.select_clear(0, "end")
                self._listbox.select_set(idx)
                self._listbox.see(idx)
                self._load_segment(idx)
                return

        # No pending left - stay on current
        self._action_status.config(text="All segments reviewed!")

    # Excel status helpers
    def _check_video_status(self, video_id: str) -> bool:
        """
        Read the Excel status for video_id before loading.
        - not 'completed' / 'validated'  → error, refuse to open
        - 'validated'                     → warn, ask if user wants to re-review
        - 'completed'                     → proceed silently
        Returns True to proceed, False to abort.
        """
        csv_path = self.catalog_dir(data_dir) / "videos_master.csv"
        if not csv_path.exists():
            return True  # can't check, proceed

        try:
            df = pd.read_csv(csv_path)
            mask = df["video_id"].astype(str) == video_id
            if not mask.any():
                messagebox.showwarning(
                    "Video not found",
                    f"Video {video_id} was not found in videos_master.csv.\n"
                    "Proceeding anyway.",
                )
                return True

            status = str(df.loc[mask.idxmax(), "status"]).strip().lower()

            if status == "validated":
                return messagebox.askyesno(
                    "Already validated",
                    f"Video {video_id} has already been fully reviewed and marked as validated.\n\n"
                    f"Do you want to review it again?",
                    icon="question",
                )

            if status not in ("completed",):
                messagebox.showerror(
                    "Not ready for review",
                    f"Video {video_id} hasn't been fully processed yet.\n"
                    f"Current status: {status}.\n\n"
                    "Please run the processing pipeline on this video first, "
                    "then come back to review it.",
                )
                return False

            return True  # completed - all good

        except Exception as e:
            messagebox.showwarning(
                "CSV check failed",
                f"Could not read CSV status:\n{e}\n\nProceeding anyway.",
            )
            return True

    def _mark_validated_videos(self):
        """
        For every video whose segments are all reviewed (none pending),
        update its Excel status to 'validated'.
        """
        # Group segments by video_id
        from collections import defaultdict
        by_video: Dict[str, List[dict]] = defaultdict(list)
        for s in self._segments:
            by_video[s["video_id"]].append(s)

        for video_id, segs in by_video.items():
            if all(s["status"] != "pending" for s in segs):
                set_excel_video_status(self.data_dir, video_id, "validated")

    # Save & Quit
    def _save_and_quit(self):
        self._stop_video()
        if self._cap:
            self._cap.release()

        # Update Excel totals for dropped segments
        if self._dropped_video_ids:
            update_excel(self.data_dir, self._dropped_video_ids, self._segments)

        pending = sum(1 for s in self._segments if s["status"] == "pending")
        if pending > 0:
            if not messagebox.askyesno(
                "Quit",
                f"{pending} segment(s) still unreviewed. Quit anyway?",
            ):
                return

        # Mark fully-reviewed videos as validated in Excel
        self._mark_validated_videos()

        self.root.destroy()

# ENTRY POINT

def main():
    parser = argparse.ArgumentParser(description="VSR Annotation Review GUI")
    parser.add_argument(
        "--data", default="./data",
        help="Path to the data directory (default: ./data)"
    )
    parser.add_argument(
        "--video", default=None,
        help="Filter to a single video ID, e.g. ro_001 or 1 (matches ro_001)"
    )
    args = parser.parse_args()

    data_dir = Path(args.data).resolve()
    if not data_dir.exists():
        print(f"Error: data directory not found: {data_dir}")
        sys.exit(1)

    # Normalise --video: "1" → "ro_001", "ro_001" → "ro_001"
    video_filter = args.video
    if video_filter is not None:
        if video_filter.isdigit():
            video_filter = f"ro_{int(video_filter):03d}"

    root = tk.Tk()

    # Set a reasonable starting size
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    w, h = min(1200, screen_w - 100), min(780, screen_h - 100)
    root.geometry(f"{w}x{h}+{(screen_w-w)//2}+{(screen_h-h)//2}")

    app = ReviewApp(root, data_dir, video_filter=video_filter)
    root.protocol("WM_DELETE_WINDOW", app._save_and_quit)
    root.mainloop()

if __name__ == "__main__":
    main()

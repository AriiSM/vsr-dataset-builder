#!/usr/bin/env python3
"""
review_transcripts_gui.py — Alegere transcript din 3 variante Whisper
=====================================================================
GUI de review rapid optimizat pentru decizie în 5-10s/segment.
Citește din dataset.db, arată cele 3 variante cu diff evidențiat,
scrie decizia înapoi în DB.

Comenzi:
    python backend/tools/review_transcripts_gui.py
    python backend/tools/review_transcripts_gui.py --tier A
    python backend/tools/review_transcripts_gui.py --sort-wer
    python backend/tools/review_transcripts_gui.py --skip-identical

Taste:
    1         — whisper_standard_mediu + avansează
    2         — whisper_dialectat_mediu + avansează
    3         — whisper_dialectat_large + avansează (dacă există)
    E         — editează manual (dialog cu toate 3 ca punct de start)
    R         — respinge segmentul
    S / →     — sare fără decizie
    ←         — înapoi
    Space     — play / pause video
"""

import argparse
import difflib
import sys
import threading
import time
from pathlib import Path
from typing import Optional, List

import cv2
import tkinter as tk
from tkinter import ttk, messagebox

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Lipsește Pillow: pip install Pillow")
    sys.exit(1)

try:
    import winsound
    _AUDIO = "winsound"
except ImportError:
    _AUDIO = None

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(_ROOT / "backend"), str(_ROOT / "backend" / "shared")]
from vsr_shared.catalog_db import CatalogDatabase  # noqa: E402
_PROC = _ROOT / "data" / "processed"


# ── DB ────────────────────────────────────────────────────────────────────────

_FIELDS = ["segment_id", "video_id", "quality_tier",
           "whisper_standard_mediu", "whisper_standard_mediu_conf",
           "whisper_dialectat_mediu", "whisper_dialectat_mediu_conf",
           "whisper_dialectat_large", "whisper_dialectat_large_conf",
           "wer_standard_vs_dialectat_mediu", "wer_mediu_vs_large"]


def load_segments(db, tier=None, skip_identical=False, sort_wer=False) -> List[dict]:
    rows = [
        [d.get(f) for f in _FIELDS]
        for d in db.segments.query(fields=_FIELDS + ["review_status"])
        if (d.get("review_status") or "") in ("", "pending")
        and (not tier or d.get("quality_tier") == tier)
    ]

    segs = []
    for r in rows:
        wer_m = r[9]
        if skip_identical and (wer_m is None or wer_m == 0.0):
            continue
        segs.append({
            "sid": r[0], "vid": r[1], "tier": r[2],
            "std":  r[3] or "", "std_conf":  r[4] or 0.0,
            "dmed": r[5] or "", "dmed_conf": r[6] or 0.0,
            "dlrg": r[7] or "", "dlrg_conf": r[8] or 0.0,
            "wer_m": wer_m, "wer_l": r[10],
            "done": False,
        })

    if sort_wer:
        segs.sort(key=lambda s: (
            0 if (s["wer_m"] or 0) > 0.05 else 1,
            -(s["wer_m"] or 0),
        ))
    else:
        segs.sort(key=lambda s: (s["tier"] or "Z", s["sid"]))

    return segs


def save_decision(db, sid: str, text: str, source: str, rejected: bool = False):
    status = "rejected" if rejected else "approved"
    db.segments.update_fields(sid, {
        "review_status": status,
        "transcript_edited": None if rejected else text,
        "transcript_source": None if rejected else source})


# ── Diff evidențiat ───────────────────────────────────────────────────────────

def word_diff(reference: str, variant: str):
    """Returnează listă de (cuvânt, tag) unde tag e 'same' sau 'changed'."""
    ref = reference.upper().split()
    var = variant.upper().split()
    result = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, var, autojunk=False).get_opcodes():
        if tag == "equal":
            for w in var[j1:j2]:
                result.append((w, "same"))
        elif tag in ("replace", "insert"):
            for w in var[j1:j2]:
                result.append((w, "changed"))
    return result


def wer_color(wer):
    if wer is None:
        return "#888888"
    if wer == 0.0:
        return "#27ae60"
    if wer <= 0.05:
        return "#2ecc71"
    if wer <= 0.20:
        return "#f39c12"
    if wer <= 0.40:
        return "#e67e22"
    return "#e74c3c"


def wer_label(wer):
    if wer is None:
        return "—"
    if wer == 0.0:
        return "0.00  (identic)"
    if wer <= 0.05:
        return f"{wer:.2f}  (aproape identic)"
    if wer <= 0.20:
        return f"{wer:.2f}  ✓ dialect capturat"
    if wer <= 0.40:
        return f"{wer:.2f}  schimbări mari"
    return f"{wer:.2f}  ⚠ grav"


# ── Dialog editare manuală ────────────────────────────────────────────────────

class EditDialog(tk.Toplevel):
    def __init__(self, parent, seg: dict):
        super().__init__(parent)
        self.title("Editare manuală")
        self.result = None
        self.resizable(True, False)
        self.grab_set()

        tk.Label(self, text="Editează transcriptul final:", font=("", 10, "bold")).pack(
            padx=12, pady=(12, 4), anchor="w"
        )

        self._entry = tk.Text(self, height=3, font=("Consolas", 11), wrap="word")
        self._entry.insert("1.0", seg["std"] or seg["dmed"] or seg["dlrg"] or "")
        self._entry.pack(fill="x", padx=12, pady=4)

        btn_row = tk.Frame(self)
        btn_row.pack(fill="x", padx=12, pady=2)
        tk.Label(btn_row, text="Completează din:", font=("", 8)).pack(side="left")
        for label, text in [
            ("Standard", seg["std"]),
            ("Dial.Med", seg["dmed"]),
            ("Dial.Large", seg["dlrg"]),
        ]:
            if text:
                tk.Button(
                    btn_row, text=f"← {label}", font=("", 8),
                    command=lambda t=text: self._fill(t),
                ).pack(side="left", padx=2)

        bot = tk.Frame(self)
        bot.pack(fill="x", padx=12, pady=(4, 12))
        tk.Button(bot, text="Salvează", bg="#27ae60", fg="white",
                  command=self._save).pack(side="right", padx=4)
        tk.Button(bot, text="Anulează", command=self.destroy).pack(side="right", padx=4)

        self.geometry("600x220")

    def _fill(self, text: str):
        self._entry.delete("1.0", "end")
        self._entry.insert("1.0", text)

    def _save(self):
        self.result = self._entry.get("1.0", "end").strip()
        self.destroy()


# ── Aplicația principală ──────────────────────────────────────────────────────

class ReviewApp:
    VIDEO_W = 420
    VIDEO_H = 300

    def __init__(self, root: tk.Tk, args):
        self.root = root
        self.root.title("VSR — Review Transcrieri")
        self.root.minsize(1300, 680)

        self._conn = CatalogDatabase()

        self._segs = load_segments(
            self._conn,
            tier=args.tier,
            skip_identical=args.skip_identical,
            sort_wer=args.sort_wer,
        )
        self._idx = 0

        # Playback
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps = 25.0
        self._total_frames = 0
        self._playing = False
        self._after_id = None
        self._play_start_time = 0.0
        self._play_start_frame = 0
        self._audio_file: Optional[Path] = None

        self._build_ui()
        self._bind_keys()

        if self._segs:
            self._load(0)
        else:
            messagebox.showinfo("Gata", "Nu există segmente de revizuit cu filtrele selectate.")

    # ── Construire UI ─────────────────────────────────────────────────────────

    def _build_ui(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        paned = ttk.PanedWindow(self.root, orient="horizontal")
        paned.grid(row=0, column=0, sticky="nsew")

        # Coloana 1: lista segmente
        left = tk.Frame(paned, width=210, bg="#f0f0f0")
        left.pack_propagate(False)
        self._build_list(left)
        paned.add(left, weight=0)

        # Coloana 2: video player
        center = tk.Frame(paned, bg="#1a1a1a")
        self._build_video(center)
        paned.add(center, weight=1)

        # Coloana 3: panoul de transcrieri
        right = tk.Frame(paned, width=500, bg="#fafafa")
        right.pack_propagate(False)
        self._build_transcripts(right)
        paned.add(right, weight=0)

        # Bara de status jos
        bar = tk.Frame(self.root, bg="#1e1e1e", height=38)
        bar.grid(row=1, column=0, sticky="ew")
        bar.grid_propagate(False)

        self._status = tk.Label(bar, text="", bg="#1e1e1e", fg="#ccc", font=("Consolas", 9))
        self._status.pack(side="left", padx=12, pady=10)

        keys = "1=Standard  2=Dial.Med  3=Dial.Large  E=Editează  R=Respinge  S/→=Sară  ←=Înapoi  Space=Play"
        tk.Label(bar, text=keys, bg="#1e1e1e", fg="#555", font=("", 8)).pack(
            side="right", padx=12, pady=10
        )

    def _build_list(self, parent):
        self._progress = tk.Label(parent, text="", bg="#f0f0f0", font=("", 9, "bold"))
        self._progress.pack(pady=(8, 2))

        frame = tk.Frame(parent, bg="#f0f0f0")
        frame.pack(fill="both", expand=True, padx=4, pady=4)

        sb = ttk.Scrollbar(frame)
        self._listbox = tk.Listbox(
            frame, yscrollcommand=sb.set, font=("Consolas", 8),
            bg="white", selectbackground="#0078d7", selectforeground="white",
            activestyle="dotbox",
        )
        sb.config(command=self._listbox.yview)
        self._listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._listbox.bind("<<ListboxSelect>>", self._on_list_select)

        for s in self._segs:
            self._listbox.insert("end", f"○ {s['sid']}")
            self._listbox.itemconfig("end", fg=wer_color(s["wer_m"]))

        self._refresh_progress()

    def _build_video(self, parent):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

        cf = tk.Frame(parent, bg="#1a1a1a")
        cf.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        self._canvas = tk.Canvas(cf, width=self.VIDEO_W, height=self.VIDEO_H,
                                  bg="#1a1a1a", highlightthickness=0)
        self._canvas.pack(expand=True)
        self._canvas.create_text(self.VIDEO_W // 2, self.VIDEO_H // 2,
                                  text="← Selectează segment", fill="#555", font=("", 11))

        ctrl = tk.Frame(parent, bg="#222")
        ctrl.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        self._seek_var = tk.DoubleVar()
        self._seek_bar = ttk.Scale(ctrl, from_=0, to=100, orient="horizontal",
                                    variable=self._seek_var, command=self._on_seek)
        self._seek_bar.pack(fill="x", padx=8, pady=4)
        self._time_lbl = tk.Label(ctrl, text="0.0s / 0.0s", bg="#222", fg="#aaa",
                                   font=("Consolas", 9))
        self._time_lbl.pack()

        self._play_btn = tk.Button(parent, text="▶  Play", bg="#0078d7", fg="white",
                                    relief="flat", font=("", 10), command=self._toggle_play)
        self._play_btn.grid(row=2, column=0, pady=6)

        self._seg_lbl = tk.Label(parent, text="", bg="#1a1a1a", fg="#888",
                                  font=("Consolas", 8))
        self._seg_lbl.grid(row=3, column=0, pady=(0, 8))

    def _build_transcripts(self, parent):
        parent.columnconfigure(0, weight=1)

        tk.Label(parent, text="Alege transcriptul corect", bg="#fafafa",
                 font=("", 11, "bold")).pack(pady=(8, 2))

        self._t_texts: List[tk.Text] = []
        self._t_wer: List[tk.Label] = []

        variants = [
            ("1", "Standard (medium)",   "#27ae60"),
            ("2", "Dialectal (medium)",  "#2980b9"),
            ("3", "Dialectal (large)",   "#8e44ad"),
        ]

        for key, title, color in variants:
            frame = tk.LabelFrame(
                parent, text=f"[{key}]  {title}", bg="#fafafa",
                font=("", 9, "bold"), fg=color, padx=6, pady=3,
            )
            frame.pack(fill="x", padx=8, pady=3)

            wlbl = tk.Label(frame, text="", bg="#fafafa", font=("", 8), fg="#666")
            wlbl.pack(anchor="w")

            txt = tk.Text(frame, height=2, wrap="word", font=("Consolas", 10),
                          bg="#f8f8f8", relief="solid", bd=1, state="disabled",
                          cursor="arrow")
            txt.pack(fill="x")
            txt.tag_config("same",    foreground="#1a1a1a")
            txt.tag_config("changed", foreground="#c0392b", font=("Consolas", 10, "bold"))
            txt.tag_config("na",      foreground="#bbb",    font=("Consolas", 10, "italic"))

            self._t_texts.append(txt)
            self._t_wer.append(wlbl)

        # Buton editare
        tk.Button(parent, text="[E]  Editează manual",
                  bg="#e67e22", fg="white", relief="flat", font=("", 10),
                  command=self._action_edit).pack(fill="x", padx=8, pady=(6, 2))

        # Buton respingere
        tk.Button(parent, text="[R]  Respinge segment",
                  bg="#c0392b", fg="white", relief="flat", font=("", 10),
                  command=self._action_reject).pack(fill="x", padx=8, pady=2)

    def _bind_keys(self):
        for widget in (self.root,):
            widget.bind("<Key-1>", lambda e: self._action_choose(0))
            widget.bind("<Key-2>", lambda e: self._action_choose(1))
            widget.bind("<Key-3>", lambda e: self._action_choose(2))
            widget.bind("<e>",     lambda e: self._action_edit())
            widget.bind("<E>",     lambda e: self._action_edit())
            widget.bind("<r>",     lambda e: self._action_reject())
            widget.bind("<R>",     lambda e: self._action_reject())
            widget.bind("<s>",     lambda e: self._navigate(1))
            widget.bind("<S>",     lambda e: self._navigate(1))
            widget.bind("<space>", lambda e: self._toggle_play())
            widget.bind("<Left>",  lambda e: self._navigate(-1))
            widget.bind("<Right>", lambda e: self._navigate(1))

    # ── Încărcare segment ─────────────────────────────────────────────────────

    def _on_list_select(self, event):
        sel = self._listbox.curselection()
        if sel:
            self._load(sel[0])

    def _load(self, idx: int):
        if not self._segs:
            return
        idx = max(0, min(idx, len(self._segs) - 1))
        self._idx = idx
        self._stop_video()

        seg = self._segs[idx]

        # Sincronizează lista
        self._listbox.select_clear(0, "end")
        self._listbox.select_set(idx)
        self._listbox.see(idx)

        # Video — caută face_crop, mouth_crop, sau direct
        for subdir in ("face_crop", "mouth_crop", ""):
            p = _PROC / seg["vid"] / subdir / f"{seg['sid']}.mp4" if subdir \
                else _PROC / seg["vid"] / f"{seg['sid']}.mp4"
            if p.exists():
                if self._cap:
                    self._cap.release()
                self._cap = cv2.VideoCapture(str(p))
                self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
                self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self._seek_bar.config(to=max(1, self._total_frames - 1))
                self._seek_var.set(0)
                self._show_frame(0)
                break

        # Audio direct din WAV
        self._stop_audio()
        wav = _PROC / seg["vid"] / "audio" / f"{seg['sid']}.wav"
        self._audio_file = wav if wav.exists() else None

        # Info segment
        dur = self._total_frames / self._fps if self._fps else 0
        wer_m = f"WER(std↔dmed)={seg['wer_m']:.2f}" if seg["wer_m"] is not None else ""
        wer_l = f"  WER(dmed↔dlrg)={seg['wer_l']:.2f}" if seg["wer_l"] is not None else ""
        self._seg_lbl.config(text=f"{seg['sid']}  tier={seg['tier']}  {dur:.1f}s  {wer_m}{wer_l}")

        # Afișează cele 3 transcrieri
        self._show_transcripts(seg)

        # Status bar
        self._status.config(text=f"Segment {idx+1}/{len(self._segs)}: {seg['sid']}")

        # Pornește redarea automat
        self.root.after(80, self._auto_play)

    def _show_transcripts(self, seg: dict):
        std = seg["std"]
        entries = [
            (seg["std"],  seg["std_conf"],  None,        True),   # standard — nu diff față de el însuși
            (seg["dmed"], seg["dmed_conf"], seg["wer_m"], False),
            (seg["dlrg"], seg["dlrg_conf"], seg["wer_l"], False),
        ]

        for i, (text, conf, wer, is_ref) in enumerate(entries):
            txt = self._t_texts[i]
            wlbl = self._t_wer[i]

            txt.config(state="normal")
            txt.delete("1.0", "end")

            if not text:
                txt.insert("end", "(indisponibil)", "na")
                wlbl.config(text="—", fg="#bbb")
            elif is_ref:
                txt.insert("end", text, "same")
                wlbl.config(text=f"conf: {conf:.2f}" if conf else "", fg="#555")
            else:
                for word, tag in word_diff(std, text):
                    txt.insert("end", word + " ", tag)
                conf_str = f"conf: {conf:.2f}  " if conf else ""
                wlbl.config(
                    text=f"{conf_str}WER: {wer_label(wer)}",
                    fg=wer_color(wer),
                )

            txt.config(state="disabled")

    # ── Acțiuni ───────────────────────────────────────────────────────────────

    def _action_choose(self, variant: int):
        if not self._segs:
            return
        seg = self._segs[self._idx]
        sources = [
            (seg["std"],  "standard_mediu"),
            (seg["dmed"], "dialectat_mediu"),
            (seg["dlrg"], "dialectat_large"),
        ]
        text, source = sources[variant]
        if not text:
            self._status.config(text=f"⚠ Varianta {variant+1} indisponibilă.")
            return
        save_decision(self._conn, seg["sid"], text, source)
        self._mark_done(self._idx)
        self._advance()

    def _action_edit(self):
        if not self._segs:
            return
        seg = self._segs[self._idx]
        dlg = EditDialog(self.root, seg)
        self.root.wait_window(dlg)
        if dlg.result is None:
            return
        save_decision(self._conn, seg["sid"], dlg.result, "manual")
        self._mark_done(self._idx)
        self._advance()

    def _action_reject(self):
        if not self._segs:
            return
        seg = self._segs[self._idx]
        save_decision(self._conn, seg["sid"], "", "rejected", rejected=True)
        self._mark_done(self._idx, rejected=True)
        self._advance()

    def _navigate(self, delta: int):
        new = self._idx + delta
        if 0 <= new < len(self._segs):
            self._load(new)

    def _advance(self):
        """Sare la primul segment ne-revizuit următor."""
        n = len(self._segs)
        for offset in range(1, n + 1):
            idx = (self._idx + offset) % n
            if not self._segs[idx]["done"]:
                self._load(idx)
                return
        self._status.config(text="✓ Toate segmentele au fost revizuite!")

    def _mark_done(self, idx: int, rejected: bool = False):
        self._segs[idx]["done"] = True
        icon = "✗" if rejected else "✓"
        color = "#c0392b" if rejected else "#27ae60"
        self._listbox.delete(idx)
        self._listbox.insert(idx, f"{icon} {self._segs[idx]['sid']}")
        self._listbox.itemconfig(idx, fg=color)
        self._refresh_progress()

    def _refresh_progress(self):
        done = sum(1 for s in self._segs if s["done"])
        total = len(self._segs)
        pct = int(100 * done / total) if total else 0
        self._progress.config(text=f"{done}/{total}  ({pct}%)")

    # ── Video ─────────────────────────────────────────────────────────────────

    def _show_frame(self, frame_no: int):
        if self._cap is None:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = self._cap.read()
        if not ret:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(self.VIDEO_W / w, self.VIDEO_H / h)
        nw, nh = int(w * scale), int(h * scale)
        rgb = cv2.resize(rgb, (nw, nh))
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self._canvas.delete("all")
        self._canvas.create_image(
            (self.VIDEO_W - nw) // 2, (self.VIDEO_H - nh) // 2,
            anchor="nw", image=self._photo,
        )
        pos = frame_no / self._fps
        tot = self._total_frames / self._fps
        self._time_lbl.config(text=f"{pos:.1f}s / {tot:.1f}s")
        self._seek_var.set(frame_no)

    def _auto_play(self):
        if not self._playing and self._cap is not None:
            self._toggle_play()

    def _toggle_play(self):
        if self._playing:
            self._stop_video()
        else:
            self._playing = True
            self._play_btn.config(text="⏸  Pause")
            if self._cap:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._play_start_frame = 0
            self._play_start_time = time.perf_counter()
            self._play_audio()
            self._play_loop()

    def _play_loop(self):
        if not self._playing or self._cap is None:
            return
        elapsed = time.perf_counter() - self._play_start_time
        target = self._play_start_frame + int(elapsed * self._fps)
        if target >= self._total_frames:
            self._stop_video()
            self._show_frame(0)
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ret, frame = self._cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            scale = min(self.VIDEO_W / w, self.VIDEO_H / h)
            nw, nh = int(w * scale), int(h * scale)
            rgb = cv2.resize(rgb, (nw, nh))
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
            self._canvas.delete("all")
            self._canvas.create_image(
                (self.VIDEO_W - nw) // 2, (self.VIDEO_H - nh) // 2,
                anchor="nw", image=self._photo,
            )
            self._time_lbl.config(text=f"{target/self._fps:.1f}s / {self._total_frames/self._fps:.1f}s")
            self._seek_var.set(target)
        self._after_id = self.root.after(16, self._play_loop)

    def _stop_video(self):
        self._playing = False
        if hasattr(self, "_play_btn"):
            self._play_btn.config(text="▶  Play")
        if self._after_id:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        self._stop_audio()

    def _on_seek(self, val):
        if self._cap and not self._playing:
            self._show_frame(int(float(val)))

    def _play_audio(self):
        if _AUDIO != "winsound" or self._audio_file is None:
            return
        threading.Thread(
            target=winsound.PlaySound,
            args=(str(self._audio_file), winsound.SND_FILENAME | winsound.SND_ASYNC),
            daemon=True,
        ).start()

    def _stop_audio(self):
        if _AUDIO == "winsound":
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    # ── Ieșire ────────────────────────────────────────────────────────────────

    def on_close(self):
        self._stop_video()
        if self._cap:
            self._cap.release()
        self.root.destroy()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Review transcrieri — alege din 3 variante Whisper")
    p.add_argument("--tier", choices=["A", "B", "C"], help="Filtrează după tier calitate")
    p.add_argument("--skip-identical", action="store_true",
                   help="Sare segmentele cu WER=0 (standard == dialectal)")
    p.add_argument("--sort-wer", action="store_true",
                   help="Sortează după WER descrescător (schimbări mari primele)")
    args = p.parse_args()

    root = tk.Tk()
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = min(1440, sw - 60), min(860, sh - 60)
    root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 2}")

    app = ReviewApp(root, args)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()

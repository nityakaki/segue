"""
Segue demo: analyse two songs, find the best transition, render it.

Usage (from the repo's top folder):
    python analysis/run_demo.py data/songs/songA.mp3 data/songs/songB.mp3

Writes to outputs/:
    analysis.png         energy, phrases and the chosen moment for both songs
    plain_fade.wav       the baseline: a simple 10-second crossfade
    segue_mix.wav        Segue's beat-matched bass-swap transition
    blind/clip_1.wav, blind/clip_2.wav, blind/answer.txt   for a blind listening test
"""

import os
import random
import shutil
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf

from features import analyze
from transition import (camelot_compatibility, explain, render_plain_fade,
                        render_segue_mix, score_candidates, tempo_ratio)

warnings.filterwarnings("ignore", category=FutureWarning)

OUT = "outputs"


def print_song(label, s):
    print(f"Song {label}: {os.path.basename(s['path'])}")
    print(f"  BPM {s['bpm']:.1f} | key {s['key']} ({s['camelot']}) | "
          f"{len(s['bar_starts'])} bars | {len(s['phrases'])} phrases | {s['duration']:.0f}s")


def plot(a, b, best, path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.5), sharex=False)
    for ax, s, mark, label in ((axes[0], a, best["a_out"], "A: leave here"),
                               (axes[1], b, best["b_in"], "B: enter here")):
        ax.plot(s["bar_starts"], s["bar_energy"], color="#7c5cff", lw=1.8, label="energy per bar")
        for p in s["phrases"]:
            ax.axvline(p, color="#bbbbbb", lw=0.7)
        ax.axvline(mark, color="#ff4d6d", lw=2.5, label=label)
        ax.set_title(f"{os.path.basename(s['path'])}  |  {s['bpm']:.0f} BPM  |  {s['camelot']}")
        ax.set_ylabel("energy")
        ax.legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("time (s)   grey lines = phrase starts (every 8 bars)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(path_a, path_b):
    os.makedirs(os.path.join(OUT, "blind"), exist_ok=True)
    print("Analysing... (about 20-40 seconds per song)\n")
    a, b = analyze(path_a), analyze(path_b)
    print_song("A", a)
    print_song("B", b)

    key_score = camelot_compatibility(a["camelot"], b["camelot"])
    stretch = (tempo_ratio(a["bpm"], b["bpm"]) - 1) * 100
    print(f"\nKey compatibility: {key_score} (1.0 = same key)")
    print(f"Tempo change needed for B: {stretch:+.1f}%")

    cands = score_candidates(a, b)
    if not cands:
        sys.exit("No valid transition points found. Try longer songs (full tracks, not previews).")
    print(f"\nScored {len(cands)} candidate transitions. Top 3:")
    for i, c in enumerate(cands[:3], 1):
        print(f"  {i}. score {c['score']}  {explain(c)}")
    best = cands[0]

    plot(a, b, best, os.path.join(OUT, "analysis.png"))
    sr = a["sr"]
    sf.write(os.path.join(OUT, "plain_fade.wav"), render_plain_fade(a, b), sr)
    sf.write(os.path.join(OUT, "segue_mix.wav"), render_segue_mix(a, b, best), sr)

    # Blind test: shuffle which file is clip 1 and clip 2
    files = ["plain_fade.wav", "segue_mix.wav"]
    random.shuffle(files)
    for i, f in enumerate(files, 1):
        shutil.copy(os.path.join(OUT, f), os.path.join(OUT, "blind", f"clip_{i}.wav"))
    with open(os.path.join(OUT, "blind", "answer.txt"), "w") as fh:
        fh.write(f"clip_1 = {files[0]}\nclip_2 = {files[1]}\n")

    print(f"\nSaved to {OUT}/: analysis.png, plain_fade.wav, segue_mix.wav, blind/")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python analysis/run_demo.py <songA> <songB>")
    main(sys.argv[1], sys.argv[2])

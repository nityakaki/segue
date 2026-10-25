"""
Check Segue's BPM and key detection against the real values.

1. Fill in data/ground_truth.csv: file,bpm,camelot  (look songs up on Beatport or Tunebat)
2. Run:  python analysis/check_accuracy.py
"""

import csv
import warnings

from features import get_bpm_and_beats, get_key_camelot, load_audio

warnings.filterwarnings("ignore")


def bpm_correct(found, real):
    """Correct if within 4% of the real BPM, or of half/double it
    (beat trackers often lock onto half or double time)."""
    return any(abs(found - real * k) / (real * k) <= 0.04 for k in (0.5, 1, 2))


rows = list(csv.DictReader(open("data/ground_truth.csv")))
bpm_hits = key_hits = 0
print(f"{'song':40} {'bpm real/found':>16} {'key real/found':>16}")
for r in rows:
    y, sr = load_audio(r["file"])
    bpm, _ = get_bpm_and_beats(y, sr)
    _, camelot = get_key_camelot(y, sr)
    b_ok = bpm_correct(bpm, float(r["bpm"]))
    k_ok = camelot == r["camelot"].strip().upper()
    bpm_hits += b_ok
    key_hits += k_ok
    print(f"{r['file'][-40:]:40} {r['bpm']:>6} / {bpm:5.1f} {'ok' if b_ok else 'X':>2}"
          f" {r['camelot']:>6} / {camelot:>4} {'ok' if k_ok else 'X':>2}")

n = len(rows)
print(f"\nBPM correct: {bpm_hits}/{n}   Key correct: {key_hits}/{n}")

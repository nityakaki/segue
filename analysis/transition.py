"""
Finding and rendering a transition between two analysed songs.

  score_candidates   -> rank every (A out-point, B in-point) pair with rules
  render_plain_fade  -> the baseline: a simple 10-second crossfade
  render_segue_mix   -> tempo-matched, beat-aligned bass swap
"""

import numpy as np
import librosa
from scipy.signal import butter, sosfiltfilt

BARS_PER_TRANSITION = 8


# ---------- Compatibility between the two songs ----------

def camelot_compatibility(code_a, code_b):
    """1.0 = same key, 0.9 = neighbour on the wheel, 0.8 = relative major/minor."""
    num_a, let_a = int(code_a[:-1]), code_a[-1]
    num_b, let_b = int(code_b[:-1]), code_b[-1]
    step = min((num_a - num_b) % 12, (num_b - num_a) % 12)
    if step == 0 and let_a == let_b:
        return 1.0
    if step == 1 and let_a == let_b:
        return 0.9
    if step == 0:
        return 0.8
    if step == 2 and let_a == let_b:
        return 0.5
    return 0.2


def tempo_ratio(bpm_a, bpm_b):
    """How much to speed B up (>1) or slow it down (<1) to match A.

    Beat trackers sometimes report half or double the real tempo, so we
    fold the ratio into the range 0.67 to 1.5.
    """
    ratio = bpm_a / bpm_b
    while ratio > 1.5:
        ratio /= 2
    while ratio < 0.67:
        ratio *= 2
    return ratio


# ---------- Scoring candidate transition points ----------

def _mean_energy(song, start_time, n_bars):
    idx = np.searchsorted(song["bar_starts"], start_time)
    return float(song["bar_energy"][idx:idx + n_bars].mean())


def score_candidates(a, b, n_bars=BARS_PER_TRANSITION):
    """Score every pair of (phrase start late in A, phrase start early in B).

    Rules, each from 0 to 1:
      energy_match : the 8 bars we leave in A feel as intense as the 8 we enter in B
      late_in_a    : leave A near its end (its outro), not halfway through
      early_in_b   : enter B near its start (its intro)
    Returns a list of candidates sorted best first.
    """
    trans_sec_a = n_bars * 4 * 60 / a["bpm"]
    trans_sec_b = n_bars * 4 * 60 / b["bpm"]
    outs = [t for t in a["phrases"] if a["duration"] * 0.5 <= t <= a["duration"] - trans_sec_a - 1]
    ins = [t for t in b["phrases"] if t <= b["duration"] * 0.4 and t + trans_sec_b + 15 < b["duration"]]

    candidates = []
    for t_out in outs:
        for t_in in ins:
            e_a = _mean_energy(a, t_out, n_bars)
            e_b = _mean_energy(b, t_in, n_bars)
            energy_match = 1 - abs(e_a - e_b)
            late_in_a = t_out / a["duration"]
            early_in_b = 1 - t_in / (b["duration"] * 0.4)
            score = 0.5 * energy_match + 0.25 * late_in_a + 0.25 * early_in_b
            candidates.append({
                "a_out": float(t_out), "b_in": float(t_in), "score": round(score, 3),
                "energy_a": round(e_a, 2), "energy_b": round(e_b, 2),
                "energy_match": round(energy_match, 2),
            })
    return sorted(candidates, key=lambda c: c["score"], reverse=True)


def explain(c):
    """One sentence a learner can read: why this moment."""
    return (f"Leave A at {c['a_out']:.1f}s and bring in B from {c['b_in']:.1f}s: "
            f"energy {c['energy_a']} -> {c['energy_b']} (match {c['energy_match']}), "
            f"both on a phrase start.")


# ---------- Rendering audio ----------

def _rms_normalise(y, target=0.1):
    return y * (target / (np.sqrt(np.mean(y ** 2)) + 1e-9))


def _split_bass(y, sr, cutoff=200):
    """Split audio into bass (below 200 Hz: kick and bassline) and everything else."""
    sos = butter(4, cutoff, btype="low", fs=sr, output="sos")
    low = sosfiltfilt(sos, y)
    return low, y - low


def render_plain_fade(a, b, fade_sec=10, lead_sec=15, tail_sec=15):
    """Baseline: fade A out over its last 10 seconds while B fades in from its start."""
    sr = a["sr"]
    ya, yb = _rms_normalise(a["y"]), _rms_normalise(b["y"])
    fade_n = int(fade_sec * sr)
    lead = ya[-(fade_n + int(lead_sec * sr)):-fade_n]
    a_fade = ya[-fade_n:]
    b_fade = yb[:fade_n]
    ramp = np.linspace(0, 1, fade_n)
    mix = a_fade * (1 - ramp) + b_fade * ramp
    tail = yb[fade_n:fade_n + int(tail_sec * sr)]
    out = np.concatenate([lead, mix, tail])
    return out / (np.abs(out).max() + 1e-9) * 0.95


def render_segue_mix(a, b, cand, n_bars=BARS_PER_TRANSITION, lead_sec=15, tail_sec=15):
    """Segue's transition:
    1. Time-stretch B so its tempo matches A.
    2. Start B exactly on A's phrase start, so the beats line up.
    3. Over 8 bars, fade A's highs out and B's highs in (equal-power curves).
    4. Halfway through, swap the bass: A's bass drops out as B's comes in,
       so two kick drums and basslines never fight each other.
    """
    sr = a["sr"]
    ratio = tempo_ratio(a["bpm"], b["bpm"])
    trans_sec = n_bars * 4 * 60 / a["bpm"]
    trans_n = int(trans_sec * sr)

    # A: some lead-in, then the transition window
    a_out_n = int(cand["a_out"] * sr)
    lead = a["y"][max(0, a_out_n - int(lead_sec * sr)):a_out_n]
    a_win = a["y"][a_out_n:a_out_n + trans_n]

    # B: cut from its in-point, stretch to A's tempo
    b_in_n = int(cand["b_in"] * sr)
    need_n = int((trans_sec + tail_sec) * sr * ratio) + sr
    b_seg = librosa.effects.time_stretch(b["y"][b_in_n:b_in_n + need_n], rate=ratio)
    b_win, tail = b_seg[:trans_n], b_seg[trans_n:trans_n + int(tail_sec * sr)]

    # Match loudness so neither song jumps out
    lead, a_win = _rms_normalise(lead), _rms_normalise(a_win)
    b_win, tail = _rms_normalise(b_win), _rms_normalise(tail)
    n = min(len(a_win), len(b_win))
    a_win, b_win = a_win[:n], b_win[:n]

    a_low, a_high = _split_bass(a_win, sr)
    b_low, b_high = _split_bass(b_win, sr)

    t = np.linspace(0, 1, n)
    fade_out, fade_in = np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
    # Bass swap at the halfway point, with a one-beat ramp so it doesn't click
    beat = 60 / a["bpm"] / trans_sec
    swap = np.clip((t - 0.5) / beat + 0.5, 0, 1)

    mix = a_high * fade_out + b_high * fade_in + a_low * (1 - swap) + b_low * swap
    out = np.concatenate([lead, mix, tail])
    return out / (np.abs(out).max() + 1e-9) * 0.95

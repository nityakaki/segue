"""
Song analysis for Segue.

Each function answers one question a DJ asks about a song:
  - How fast is it, and where are the beats?   -> get_bpm_and_beats
  - What key is it in (as a Camelot code)?      -> get_key_camelot
  - How loud / intense is each bar?             -> get_energy_per_bar
  - Where do the musical phrases start?         -> get_phrase_starts
"""

import numpy as np
import librosa

SR = 22050  # sample rate: 22,050 samples per second is plenty for analysis


def load_audio(path, sr=SR):
    """Load a song as a mono numpy array of samples."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def get_bpm_and_beats(y, sr):
    """Return the tempo (BPM) and the time in seconds of every beat.

    librosa finds 'onsets' (moments where sound suddenly gets louder, like a
    kick drum), then finds the steady pulse that best lines up with them.
    """
    tempo, beat_times = librosa.beat.beat_track(y=y, sr=sr, units="time")
    bpm = float(np.atleast_1d(tempo)[0])
    return bpm, beat_times


# Krumhansl-Schmuckler key profiles: how strongly each of the 12 notes
# "belongs" in a major or minor key, measured from listening experiments.
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Camelot wheel: DJs use these codes because neighbours (8A -> 9A, 8A -> 8B)
# always sound good together.
CAMELOT_MAJOR = {"C": "8B", "G": "9B", "D": "10B", "A": "11B", "E": "12B", "B": "1B",
                 "F#": "2B", "C#": "3B", "G#": "4B", "D#": "5B", "A#": "6B", "F": "7B"}
CAMELOT_MINOR = {"A": "8A", "E": "9A", "B": "10A", "F#": "11A", "C#": "12A", "G#": "1A",
                 "D#": "2A", "A#": "3A", "F": "4A", "C": "5A", "G": "6A", "D": "7A"}


def get_key_camelot(y, sr):
    """Estimate the song's key and return (key name, Camelot code).

    1. A chromagram measures how much of each of the 12 notes is playing.
    2. We average it over the whole song.
    3. We compare that average against all 24 keys (12 major + 12 minor)
       and pick the key whose profile matches best (highest correlation).
    """
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    best_score, best_key = -2.0, None
    for i, note in enumerate(NOTES):
        for profile, mode in ((MAJOR_PROFILE, "major"), (MINOR_PROFILE, "minor")):
            score = np.corrcoef(chroma, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, best_key = score, (note, mode)
    note, mode = best_key
    camelot = CAMELOT_MAJOR[note] if mode == "major" else CAMELOT_MINOR[note]
    return f"{note} {mode}", camelot


def get_downbeat_offset(y, sr, beat_times):
    """Guess which beat is '1' of the bar (0, 1, 2 or 3).

    In most pop and dance music the first beat of each bar is the strongest,
    so we try all four options and keep the one with the loudest hits.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = librosa.time_to_frames(beat_times, sr=sr)
    frames = np.clip(frames, 0, len(onset_env) - 1)
    strength = onset_env[frames]
    scores = [strength[k::4].mean() if len(strength[k::4]) else 0 for k in range(4)]
    return int(np.argmax(scores))


def get_energy_per_bar(y, sr, beat_times, offset=0):
    """Return (bar start times, energy of each bar from 0 to 1).

    Energy = RMS loudness, averaged over each bar of 4 beats, then scaled so
    the loudest bar is 1.0. Drops are near 1, intros and breakdowns are lower.
    """
    rms = librosa.feature.rms(y=y)[0]
    rms_times = librosa.times_like(rms, sr=sr)
    bar_starts = beat_times[offset::4]
    energies = []
    for i, start in enumerate(bar_starts):
        end = bar_starts[i + 1] if i + 1 < len(bar_starts) else rms_times[-1]
        mask = (rms_times >= start) & (rms_times < end)
        energies.append(rms[mask].mean() if mask.any() else 0.0)
    energies = np.array(energies)
    return bar_starts, energies / (energies.max() + 1e-9)


def get_phrase_starts(bar_starts, bars_per_phrase=8):
    """Return the times where phrases start.

    Most dance and pop music is built in phrases of 8 bars (32 beats).
    DJs switch songs at a phrase start so the new song's sections line up.
    """
    return bar_starts[::bars_per_phrase]


def analyze(path):
    """Run every analysis step on one song and return the results as a dict."""
    y, sr = load_audio(path)
    bpm, beats = get_bpm_and_beats(y, sr)
    key, camelot = get_key_camelot(y, sr)
    offset = get_downbeat_offset(y, sr, beats)
    bar_starts, bar_energy = get_energy_per_bar(y, sr, beats, offset)
    phrases = get_phrase_starts(bar_starts)
    return {
        "path": path, "y": y, "sr": sr, "duration": len(y) / sr,
        "bpm": bpm, "beats": beats, "key": key, "camelot": camelot,
        "bar_starts": bar_starts, "bar_energy": bar_energy, "phrases": phrases,
    }

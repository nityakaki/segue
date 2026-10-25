# Segue

An AI DJ that teaches you **when** and **how** to mix two songs, and can render the transition for you.

Segue analyses two songs (tempo, key, energy, phrases), scores every possible switch point, picks the best one, and renders a beat-matched transition. It also renders a plain crossfade so the two can be compared in a blind listening test.

## Status (midpoint, Oct 2026)

| Done | Next |
| --- | --- |
| Repo, folder structure, weekly milestones | Label ~100 transitions and train a logistic regression ranker (Week 2) |
| Song analysis: BPM, beats, key to Camelot, downbeats, energy per bar, 8-bar phrases | Filter sweep, echo out and vocal-aware (Demucs) techniques (Week 3) |
| Rule-based transition scorer with a plain-English reason | Teacher mode web app: FastAPI + React, 6 guided steps (Week 4) |
| Renderer: tempo match, beat alignment, equal-power fade, bass swap | Blind listening test, 20 song pairs, 10+ listeners (Week 5) |
| Plain-fade baseline and blind A/B clip generator | |
| Accuracy check script for BPM and key | |

## How it works

```
song A, song B
   -> features.py     BPM + beats, key (Camelot), downbeat, energy per bar, phrase starts
   -> transition.py   score (A out, B in) phrase pairs -> best moment + reason
                      render plain fade (baseline) and Segue mix
   -> outputs/        analysis.png, plain_fade.wav, segue_mix.wav, blind/clip_1, clip_2
```

## Setup (Mac)

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put two songs in `data/songs/` (MP3 or WAV work out of the box; for iTunes .m4a files run `brew install ffmpeg` first).

## Run

```
python analysis/run_demo.py data/songs/songA.mp3 data/songs/songB.mp3
python analysis/check_accuracy.py      # after filling in data/ground_truth.csv
```

## Evaluation plan

Following mentor feedback: compare Segue's rendered transition against a plain 10-second crossfade in blind listening tests on a fixed set of song pairs, and report the win rate.

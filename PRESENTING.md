# Midpoint cheat sheet

Everything you need to explain the code in about 5 minutes. You don't need to know every line; know these ideas.

## 1. The one-sentence pitch

"Segue is an AI DJ that finds the best moment to switch between two songs, teaches you how to do the transition, and can render it for you. I'm testing whether it sounds better than a plain fade with blind listening tests."

## 2. Demo order (about 4 minutes)

1. **GitHub repo**: show the folders and the Milestones page (5 weeks, Week 1 done).
2. **Run the demo** in Terminal, or show the output you saved earlier. Point at the printed BPM, key and top 3 moments.
3. **Show `outputs/analysis.png`**: purple line = energy per bar, grey lines = phrase starts, red lines = where Segue switches.
4. **Blind test**: play `blind/clip_1.wav` and `clip_2.wav`. Ask the room which sounded smoother, then reveal `answer.txt`.
5. **Next steps**: the roadmap from the blueprint.

## 3. What each file does

**features.py: understanding one song**
- `get_bpm_and_beats`: librosa finds sudden jumps in loudness (drum hits) and the steady pulse that fits them. That gives the BPM and the time of every beat.
- `get_key_camelot`: measures how much of each of the 12 notes plays across the song, compares it with the "fingerprint" of all 24 keys, and picks the closest. It then converts to a Camelot code (like 8A), which DJs use because neighbouring codes sound good together.
- `get_downbeat_offset`: in most songs beat 1 of each bar hits hardest, so it tries all four options and picks the loudest.
- `get_energy_per_bar`: average loudness of each bar, scaled from 0 to 1. Drops sit near 1, breakdowns lower.
- `get_phrase_starts`: songs are built in 8-bar phrases. DJs switch on a phrase start so sections line up.

**transition.py: picking and making the transition**
- `score_candidates`: tries every pair of (phrase start late in song A, phrase start early in song B). It scores each pair on three rules: energy match (50%), leaving A near its end (25%) and entering B near its start (25%). It returns them best first.
- `camelot_compatibility`: 1.0 for the same key, 0.9 for a neighbour on the wheel, 0.8 for relative major or minor.
- `render_plain_fade`: the baseline. A fades out over its last 10 seconds while B fades in.
- `render_segue_mix`: the real transition.
  1. Speeds song B up or slows it down to match A's tempo, without changing pitch.
  2. Starts B exactly on A's phrase start, so the beats line up.
  3. Fades A's highs out and B's highs in over 8 bars, using equal-power curves so the volume doesn't dip.
  4. Swaps the bass halfway: A's kick and bassline drop out as B's come in, so two basslines never clash. Real DJs do exactly this with the EQ knob.

**run_demo.py** runs everything and saves the plot, both mixes and a shuffled blind-test pair.
**check_accuracy.py** compares detected BPM and key with the real values from Beatport or Tunebat.

## 4. Questions you might get

- **"Where's the AI?"** "Right now the scorer is rule-based on purpose. That's my baseline. Next week I label about 100 transitions by hand and train a logistic regression model to rank the moments. Then I measure whether it beats the rules (top-3 accuracy). I chose logistic regression because its weights explain *why* a moment is good, which the Teacher mode needs."
- **"Why not Spotify?"** "Spotify doesn't give apps raw audio or allow mixing through its public API. DJ apps like djay got it through a partnership in 2025. My prototype mixes files you own, and a Spotify partnership is the scaling step."
- **"How will you know it's good?"** "Blind listening tests on 20 fixed song pairs: plain fade vs Segue, random order, 10+ listeners, and I report the win rate."
- **"What's hard?"** "Key detection. My accuracy check shows where it misses, and Essentia's key detector is the planned fix. Also songs with tempo changes or no clear beat."
- **"Why 8-bar phrases?"** "Most pop and dance music is written in 8-bar sections. Switching on a phrase start keeps the new song's chorus or drop landing where your ear expects it."

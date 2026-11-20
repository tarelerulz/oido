# Hybrid CPU + GPU split of one phrase (experiment)

A long phrase is cut at a real pause. The GPU transcribes the first part (the *head*) while a resident CPU engine transcribes
the rest (the *tail*) at the same time, and the two texts are joined. Both engines work on their own audio with their own full
pipeline, so there is no per-layer synchronisation or shared buffers.

Files:
- `hybrid_tail.py` - `HybridTail`: `split(pcm)` finds the cut, `tail(pcm)` asks a resident `tasr_cli --serve` for the tail text.
  Any CPU-side problem returns `None` and the caller transcribes the whole phrase on the GPU as before.
- `test_hybrid_tail.py` - synthetic checks of the cut rule (no models or recordings needed): `python3 hybrid/test_hybrid_tail.py`.
- `../components/tinyasr/tinyasr.c` - fix needed for this to work (see below).

## Measured (Pi 400, GPU at 800 MHz, performance governor; 2026-11-19; "wall" = time from the end of the audio to the text)

10-20 s clips, resident CPU engine (3 threads on cores 1-3, nice 19), cut at a pause about 60 % into the clip:

| clip | GPU only | hybrid | text |
|---|---|---|---|
| 10 s (two different clips) | 488 / 480 ms | 390 / 388 ms (-20 %) | same |
| 15 s | 817 ms | 501 ms (-39 %) | same |
| 20 s | 1143 ms (past the GPU path's 19 s limit, text cut off) | 737 ms | hybrid text complete |

76 recorded dictation phrases of 4.5 s or longer, timed GPU head + CPU tail in parallel, cut rule `RATIO = 0.05`:
25 phrases were split (the rest had no safe pause and stay GPU-only); on those the median wall time was 30 % lower.
Words versus the GPU-only text: 13 identical, 7 differ by one word, 5 differ by two or more. There is no ground truth, and the
differences are a mix of better and worse words (a cut can fall inside a word's context). A looser quiet threshold splits more
phrases but changes more words.

## Caveats

- **The real gain in live dictation is small.** A dictation front end that already starts transcribing after a short quiet
  and types only after a longer one hides most of the wait; the split mainly helps phrases over about 8 s (roughly 7 % of
  phrases in one person's logs). Quality in daily use was NOT yet judged.
- **Needs a pause.** No quiet 100 ms window between 35 % and 75 % of the phrase, or an estimated gain under 10 %: no split.
  Phrases under `OIDO_HYBRID_MIN` (4.5 s) are never split.
- **Pin the CPU threads correctly.** Before this change the `tinyasr` thread pool ignored the inherited `taskset` mask (main
  thread on core 0, workers on cores 1-2 whatever the mask said), so `taskset -c 1,2,3` did not keep core 0 free for the GPU
  helper. `components/tinyasr/tinyasr.c` now keeps only cores in the initial affinity mask. Checked on the Pi by listing each
  worker's affinity (`/proc/<pid>/task/*`): with `taskset -c 1,2,3` the three threads sit on cores 1, 2 and 3.
- **Don't load all four cores.** Running a benchmark that pins every core made a Pi 400 unusable for about a minute per
  keypress. Keep one core free, run at `nice 19`, and keep runs short.
- **Tuning is for one machine.** `GPU_PTS` (GPU time versus head length) and the CPU model `10 + 45 ms per second` were measured
  on a Pi 400 with the hand-written QPU matmul kernels from gpu-port-kit. Recalibrate them for another engine or board. The
  Pi 5 was not tested.
- Related ideas that were tested and rejected: splitting single matmuls or FFN blocks between CPU and GPU inside a layer
  (at most about 2 % on a whole clip, and the int8 CPU part changes the output).

Environment variables: `OIDO_HYBRID_MIN` (default 4.5 s), `OIDO_HYBRID_RATIO` (default 0.05; higher = more splits, more changed
words), `OIDO_HYBRID_DEBUG=1` (log to stdout).

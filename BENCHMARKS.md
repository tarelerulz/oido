# Benchmarks

Speed and accuracy numbers for the oido speech engine (CPU, GPU and the hybrid that uses both) and for the heart / heart-nano
voices, on a Raspberry Pi 400. **Start here**; older tables are kept at the bottom.

Every number is marked with how solid it is:

- **Measured 2026-11-22/23**: run in one session on the same board with the scripts listed under [Reproduce](#reproduce).
- **Earlier**: measured 2026-11-17 to 2026-11-21 and **not re-run** for this page.
- **Single run**: one pass over the test set. Differences under about 0.5 points of error rate or about 10 % of time are noise.

Nothing here has been judged by listening, and nothing has been run on a Pi 5 (see [Limits](#limits)).

## Contents

1. [Summary](#summary)
2. [Speed against clip length](#speed-against-clip-length)
3. [Hybrid cut rules: speed and word errors](#hybrid-cut-rules-speed-and-word-errors)
4. [Four ideas for a faster hybrid](#four-ideas-for-a-faster-hybrid)
5. [Earlier tuned hybrid numbers](#earlier-tuned-hybrid-numbers)
6. [Dot-product (sdot) builds](#dot-product-sdot-builds)
7. [heart and heart-nano](#heart-and-heart-nano)
8. [Tried and rejected](#tried-and-rejected)
9. [Limits](#limits)
10. [Reproduce](#reproduce)
11. [Older results](#older-results-this-build-vs-upstream-oido-v2-cpu-build-before-the-4x4-neon-change)

## Summary

Test: 94 clips made from 40 everyday sentences (see [test set](#test-set)), 2385 reference words, Pi 400. Mean wall time from the
end of the audio to the text; word error rate against the known text. **Measured 2026-11-23, single run.**

| Engine | Word error rate | Mean time |
|---|---|---|
| CPU only (resident engine, 3 threads) | **6.6 %** | 312 ms |
| GPU only | 6.8 % | 531 ms |
| Hybrid, live cut rule | 8.3 % | 265 ms |

- CPU-only is the most accurate and about 1.7x faster than GPU-only on this board.
- The hybrid is about 15 % faster than CPU-only and about 2x faster than GPU-only, but it costs about 1.7 points of word error rate,
  because the text is cut and joined.
- The hybrid only clearly beats CPU-only on phrases of about 9 s or more (by about 25 %). Under 9 s it is no faster and less accurate.
- Streaming the head of a phrase while the speaker is still talking would cut the wait after the speech ends to about 190 ms
  (simulated, see [four ideas](#four-ideas-for-a-faster-hybrid)).

### Test set

40 sentences from Simple English Wikipedia (CC BY-SA) and its help pages, chosen so there are no names, digits or symbols:
8 imperative "command" sentences, 12 "you can ..." instruction sentences and 20 plain everyday sentences. Each sentence was read by
the heart voice (synthetic, one voice), resampled to 16 kHz, then used alone (if 4.5 s or longer) and joined in pairs and triples
with 0.4 s of silence to make 94 clips of 4.5 to 18.5 s. Word error rate is word-level edit distance on lower-case letters.
Engines run as in live dictation: no language model, GPU path decodes greedily.

An earlier set of 39 Wikipedia sentences with many names (for example "Aldrin") gave higher errors for every engine
(CPU-only 10.7 %, GPU-only 11.3 %, hybrid 13.2 %) with the same ranking; names the voice mispronounces count as errors for all of them.

## Speed against clip length

**Measured 2026-11-22, median of 5 runs, ms.** Slices of the 11 s `samples/jfk.wav`. "Hybrid" here is the simple version: the last 40 % of
the clip goes to the CPU, the first 60 % to the GPU, no attempt to find a pause.

| Clip | CPU only (3 threads) | GPU only | Hybrid, fixed 60 % cut |
|---|---|---|---|
| 2.5 s | **71** | 134 | 145 |
| 5 s | **158** | 245 | 210 |
| 10 s | **323** | 463 | 347 |
| 15 s | 531 | 782 | **465** |

On 12 recorded dictation phrases of 5 to 18 s (not included, they are a person's voice), with the live pause-based cut: hybrid took 0.53x
the GPU-only time and 0.92x the CPU-only time (median). It beat CPU-only clearly only from about 11 s (about 26 % faster at 11.8 s and 33 % at 17.5 s).

Measured speed of each engine on its own (**2026-11-22**): CPU about 33 ms per second of audio; GPU 103 ms for 1 s, 225 ms for 4 s,
363 ms for 7 s, 470 ms for 10 s, 746 ms for 13 s, 984 ms for 17.5 s (a staircase at 2.56 s blocks). This boot's GPU was about 30 %
slower than the curve measured on 2026-11-20.

## Hybrid cut rules: speed and word errors

The hybrid cuts a phrase at one point, transcribes the head on the GPU and the tail on the CPU at the same time, and joins the texts.
Where to cut is a pluggable rule (`OIDO_HYBRID_RULE` in `oido-type.py` in the `oido-streaming` repo). **Measured 2026-11-23, single run, 94 clips.**

| Cut rule | Word error rate | Mean time | 4.5 to 9 s clips (error / time) | 9 to 19 s clips (error / time) |
|---|---|---|---|---|
| CPU only (no cut) | 6.6 % | 312 ms | 6.3 % / 202 ms | 6.7 % / 405 ms |
| GPU only (no cut) | 6.8 % | 531 ms | 6.7 % / 358 ms | 6.8 % / 676 ms |
| `live`: quiet moments between 28 % and 70 % of the phrase | 8.3 % | 265 ms | 9.4 % / 210 ms | 7.8 % / 311 ms |
| `balance`: balance point of the speed curves, anywhere | 9.9 % | 257 ms | 10.9 % / 197 ms | 9.4 % / 308 ms |
| `snap`: balance point moved to the quietest moment within 0.5 s | 8.3 % | 256 ms | 9.2 % / 193 ms | 7.9 % / 310 ms |
| balance on fresh curves, quiet moments only | 8.1 % | 259 ms | 9.4 % / 205 ms | 7.5 % / 303 ms |

Cutting in the middle of speech (`balance`) is the fastest on long clips and the least accurate. Snapping to a quiet moment recovers most of the loss.
No hybrid rule matched CPU-only accuracy.

## Four ideas for a faster hybrid

**Measured 2026-11-23, same 94 clips, single run.** Baseline: live rule 8.3 % / 265 ms.

| Idea | Word error rate | Mean time | Result |
|---|---|---|---|
| 1. Overlap each side of the cut by 1 s and stitch the two texts | 8.3 % | 301 ms | No gain. Short clips improved (9.4 to 8.3 %), long clips got worse (7.8 to 8.3 %), and it is about 36 ms slower. Stitching the per-frame outputs instead of the text is untested (the CLI prints only text). |
| 2. Start transcribing the head while the speaker is still talking | 8.3 % | **191 ms** after the speech ends | **Simulation**, not built: head and tail timed separately, the cut is assumed known 0.3 s after the pause, GPU/CPU contention while speaking is ignored. 138 ms for 4.5 to 9 s clips, 235 ms for 9 to 19 s. |
| 3. Balance the cut on freshly measured speed curves, quiet moments only | 8.1 % | 259 ms | Within noise of the live rule. |
| 4. Phrases under 9 s go to CPU-only, longer ones to the hybrid | **7.3 %** | 261 ms | A point better than the live rule at the same speed. (Under 9 s to GPU-only instead: 7.4 % / 333 ms.) |

## Earlier tuned hybrid numbers

**Earlier (2026-11-20), not re-run.** The tuned hybrid on single clips, before the GPU got slower on the later boot: 5 s clip cut at 2.5 s
141 ms (GPU only 238 ms); 10 s clip 243 ms (463 ms); 11 s `jfk.wav` 267 ms (582 ms); 15 s clip cut at 5.1 s 313 ms (781 ms). On 82 recorded
phrases the median went from 323 ms (GPU only) to 214 ms. These were not scored for word errors against known text.

The resident CPU engine took about 32.5 ms per second of audio plus 1 ms on 3 threads (exact build). The 4x4 NEON matmul, 4-row attention and
cached position terms made the CPU engine about 30 % faster than the earlier build (10 s clip about 570 to 400 ms in the one-shot CLI).
An earlier claim of 1.8x came from a build whose results were not bit-exact and was withdrawn.

## Dot-product (sdot) builds

**Earlier (2026-11-20), not re-run, Pi 5 untested.** The Pi 400's Cortex-A72 has no dot-product instructions, so these builds are for phones and the Pi 5.
oido with 4x4 sdot matmul and 4-row sdot attention, logits bit-exact against the plain build, on a Pixel 9 (3 A720 cores): 10 s clip
94.7 to 62.3 ms (1.49 to 1.56x); one A720 core 1.70 to 1.72x; one-shot CLI 1.20 to 1.55x. On the Pi 400 the same source gave 321 to 315 ms.

## heart and heart-nano

**Earlier (2026-11-17 to 2026-11-21), not re-run for this page**, text-to-speech engines (sanoTTS) running on the Pi 400 unless noted.

| Change | Before | After | Note |
|---|---|---|---|
| NEON decoder, heart-nano, 7 s of audio, one-shot CLI | 104 to 123 ms | 48 to 72 ms (decoder 72 to 82 -> 20 ms) | at most 1 LSB different |
| NEON build of the heart CLI, one sentence, whole process | 376 ms | 249 ms (engine 326 -> 150 to 184 ms) | at most 1 LSB different |
| Keep the model loaded, one sentence, heart | 250 ms | 155 to 160 ms | same audio as the one-shot CLI |
| Keep the model loaded, one sentence, heart-nano | 78 ms | 40 ms | same audio as the one-shot CLI |
| heart-nano matmuls on the GPU (QPU kernels), one sentence | 92 ms | 73 ms | also about half the CPU time |
| heart-nano GPU path, round 2 (fast WAV, threaded finish, twin buffer), long text | 176 ms | 147 ms | byte-identical except one folded layer norm |

Dot-product (sdot) builds of heart and heart-nano, measured on a phone, are 1.5 to 1.7x faster for a vectorized 8x8 int8 variant but differ
from the plain build (SNR 26 to 29 dB), so they are not the default; an FP16 variant is 1.37 to 1.64x faster at 48 to 50 dB. Only one heart-nano
sentence was listened to (it sounded the same as the base); the others were not.

## Tried and rejected

- **Splitting single matmuls between CPU and GPU** inside a layer: about 2 % on a whole clip at best, and the int8 CPU part changes the output.
- **Running CPU and GPU together slows both**: GPU 459 to 491 ms, CPU 554 to 691 ms in the same run.
- **heart-nano int8 sdot**: slower on the Pixel 9 (decoder 5.5 to 6 ms) and 27 dB SNR.
- **Converting heart-nano int8 weights to f32**: 0.72 to 0.77x (slower).
- **Hand-scheduled NEON assembly, a lock-free thread pool, a 64-row GPU kernel, smaller workgroups and several other GPU experiments**: no gain or slower; the
  GPU details are in [gpu-port-kit](https://github.com/tarelerulz/gpu-port-kit).

## Limits

- **One voice, synthetic speech.** The word error numbers use the heart TTS voice. Real voices will have higher errors for every engine, and the
  gaps between engines may change. Recorded dictation was only used for speed.
- **Single runs.** No repeats of the accuracy runs and no significance test.
- **No listening checks** of the hybrid or sdot output.
- **Pi 400 only for the oido numbers**; the Pi 5 and phone are untested or earlier-only.
- **Simulated, not built:** the streaming-head latency.
- **No language model** in the word-error runs; with it the errors would be lower.
- Clock and governor: performance governor; the GPU (V3D) clock was read back as 800 MHz (`v3d_freq=800`, `over_voltage=6` in the boot config) after these runs. The CPU was not overclocked.

## Reproduce

Build the CPU engine with the resident mode used by the hybrid, then feed it WAV paths on stdin; it prints `path<TAB>text` per line:

```sh
cd host && make tasr_cli_a72
printf '../samples/jfk.wav\n' | taskset -c 1,2,3 nice -n 19 ./tasr_cli_a72 nemo8.tnm --serve
```

The `--serve` mode was missing from this repo until the pull request that added this page. The benchmark scripts (speed against clip length,
cut rules, the four ideas), the test set and the scoring code are not in the repo yet; they are planned for a follow-up pull request.


---

## Older results: this build vs upstream oido v2 (CPU build before the 4x4 NEON change)

All numbers are measured on the hardware named below. Nothing here is a word-error benchmark.

### Test setup

- **Clips** (16 kHz, mono, 16-bit WAV):
  - `tarel40.wav`: 10.00 s
  - `movie10.wav`: 10.00 s
  - `en16.wav`: 3.85 s
- **Settings:** beam 4 with the language model on (`--beam 4 --lm <file>`).
- **Method:** each model and clip run 10 times, alternating models between runs. The median is reported, with min–max in brackets.
- **Timer:** wall-clock time around the whole `tasr_cli` process, so it includes model load and process start.
- **Models:**
  - "This build" = `nemo8.tnm` + `nemo_lm.tlm` (upstream nemo8 weights, this repo's C engine with the Pi and DOTPROD changes in `optimization_log.md`).
  - "v2a" / "v2b" = `oido_v2a.tnm` / `oido_v2b.tnm` + `oido_v2.tlm`, from `lokutor-ai/oido` v2.0.0 (int4).

### Pi 400 (Cortex-A72, NEON only, no DOTPROD)

Cores pinned with `taskset`. Build with `-O3 -march=armv8-a -mtune=cortex-a72 -DTASR_NW=8`.

| Clip (length) | Cores | This build | v2a | v2b |
|---|---|---|---|---|
| tarel40 (10.00 s) | 0–2 | **529 ms** (504–683) | 929 ms (915–951) | 757 ms (743–768) |
| movie10 (10.00 s) | 0–2 | **569 ms** (508–631) | 957 ms (945–975) | 775 ms (767–801) |
| en16 (3.85 s) | 0–2 | **216 ms** (202–235) | 374 ms (369–385) | 316 ms (311–334) |
| tarel40 (10.00 s) | 0–3, 4 threads | **527 ms** (486–663) | 925 ms (921–961) | 752 ms (745–771) |

On the Pi, this build is the fastest in wall-clock time on every clip. v2 uses less CPU time (about 1.9× less for v2a and 2.4× less for v2b, measured per process), but it does not finish sooner.

### Android phone (Pixel 9 Linux Terminal VM, ARMv8.2 + DOTPROD)

Pinned to the four fast cores (4–7), 4 threads, `TASR_THREADS=4`. Build: `-march=armv8.2-a+dotprod`.

| Clip (length) | This build | v2a | v2b |
|---|---|---|---|
| tarel40 (10.00 s) | **141 ms** (137–158) | 166 ms (129–185) | 150 ms (142–164) |

On the phone's fast cores this build is faster than both v2 models: about 6% faster than v2b and about 15% faster than v2a. The v2b gap is small, and the run ranges overlap.

#### DOTPROD and NEON-only (this build, same phone, same clip)

| Build | Median (min–max) |
|---|---|
| DOTPROD (`-march=armv8.2-a+dotprod`) | 130 ms (116–273) |
| NEON only (`-march=armv8-a`) | 161 ms (149–491) |

DOTPROD gives about 20% here. Upstream nemo8 gains more from DOTPROD (about 2.8×) because its build lacks the NEON changes in this repo.

### Things to know before comparing

- **The two engines are not the same code.** This build has the NEON and DOTPROD changes from `optimization_log.md`. The v2 numbers use the v2.0.0 tree as released, built with the same compiler flags. A same-source comparison would be needed to attribute the gap to specific changes. The upstream nemo8 build was measured to be slower than this build on the same phone (see below), so the gap is not only the model.
- **Upstream nemo8 vs this build (phone, big cores, 4 threads, tarel40):** upstream sources 260 ms median; this build 141 ms. The upstream numbers were built from the v2.0.0 source tree, not from this repo.
- **The model is not the same as v2.** v2b makes more word errors than this build's nemo8 on `movie10.wav`. No word-error rate was computed on these clips; they are too few to be a benchmark.
- **The licences differ.** This repo's code is GPL-3. The v2 weights are CC BY-NC-SA 4.0 (non-commercial). This build uses the upstream nemo8 weights.
- **Upstream phone RTFx numbers** in `docs/phone-benchmarks.md` (for example 121× on four threads) were measured on a different phone and on a different clip. They don't match the 4-fast-core timing above (about 38× for the same model on the Pixel 9 VM). The cause is not established.

### Reproduce

```sh
cd esp32/host            # or host/ in this repo
make tasr_cli_a72        # Pi 400
# or
make tasr_cli_phone      # Android, ARMv8.2 + DOTPROD

taskset -c 0-2 ./tasr_cli_a72 nemo8.tnm --lm nemo_lm.tlm --beam 4 tarel40.wav
```

Time each run with `date +%s%N` before and after the command. Use 10 runs per model and report the median.

### Raw data

The raw per-run timings and the scripts used are kept with the author's private test notes, not in this repo.

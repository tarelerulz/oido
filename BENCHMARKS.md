# Benchmarks

Speed and accuracy numbers for the oido speech engine (CPU, GPU and the hybrid that uses both) and for the heart / heart-nano
voices, on a Raspberry Pi 400. **Start here**; older tables are kept at the bottom. All times are in milliseconds (1,000 ms = 1 second).

Every number is marked with how solid it is:

- **Measured 2026-11-22/23**: run in one session on the same board with the scripts listed under [Reproduce](#reproduce).
- **Earlier**: measured 2026-11-17 to 2026-11-21 and **not re-run** for this page.
- **Single run**: one pass over the test set. Differences under about 0.5 points of error rate or about 10 % of time are noise.

Nothing here has been judged by listening, and nothing has been run on a Pi 5 (see [Limits](#limits)).

## Contents

1. [Summary](#summary)
2. [What a user waits for, end to end](#what-a-user-waits-for-end-to-end)
3. [Real recorded phrases: speed and accuracy](#real-recorded-phrases-speed-and-accuracy)
4. [Speed against clip length](#speed-against-clip-length)
5. [Hybrid cut rules: speed and word errors](#hybrid-cut-rules-speed-and-word-errors)
6. [Four ideas for a faster hybrid](#four-ideas-for-a-faster-hybrid)
7. [Other speed tests](#other-speed-tests)
8. [Earlier tuned hybrid numbers](#earlier-tuned-hybrid-numbers)
9. [Dot-product (sdot) builds](#dot-product-sdot-builds)
10. [heart and heart-nano](#heart-and-heart-nano)
11. [Tried and rejected](#tried-and-rejected)
12. [Limits](#limits)
13. [Reproduce](#reproduce)
14. [Older results](#older-results-this-build-vs-upstream-oido-v2-cpu-build-before-the-4x4-neon-change)

## Summary

- **The wait after you stop talking is about 720 to 745 ms and is mostly not the engine.** Dictation types a phrase only after a quiet period
  (600 ms by default) confirms you stopped. The engine starts early (after 250 ms of quiet) and finishes inside that wait, so for typical phrases the
  CPU, GPU and hybrid engines are within about 30 ms of each other end to end. See [end to end](#what-a-user-waits-for-end-to-end).
- **A shorter quiet rule is the biggest lever measured:** 500 ms saves about 130 ms and 400 ms about 240 ms per phrase, for about 1 and 4
  extra word errors per 364 words.
- **On engine time alone the CPU (4 threads) is the fastest and as accurate as the others** on real phrases of one speaker:
  189 ms against 326 ms (GPU) and 275 ms (hybrid) for phrases of 4,500 to 18,500 ms. The hybrid adds word errors and no speed on those phrases.
- **A stop-word check adds about 123 ms** to phrases of 3,000 ms or less (about half of the recorded phrases) unless it is started early.

Test used for the synthetic-voice numbers below: 94 clips made from 40 everyday sentences (see [test set](#test-set)), 2,385 reference words.
Mean wall time from the end of the audio to the text; word error rate against the known text. **Measured 2026-11-23, single run.**

| Engine | Word error rate | Mean time |
|---|---|---|
| CPU only (resident engine, 3 threads) | **6.6 %** | 312 ms |
| GPU only | 6.8 % | 531 ms |
| Hybrid, live cut rule | 8.3 % | 265 ms |

### Test set

40 sentences from Simple English Wikipedia (CC BY-SA) and its help pages, chosen so there are no names, digits or symbols:
8 imperative "command" sentences, 12 "you can ..." instruction sentences and 20 plain everyday sentences. Each sentence was read by
the heart voice (synthetic, one voice), resampled to 16 kHz, then used alone (if 4,500 ms or longer) and joined in pairs and triples
with 400 ms of silence to make 94 clips of 4,500 to 18,500 ms. Word error rate is word-level edit distance on lower-case letters.
Engines run as in live dictation: no language model, GPU path decodes greedily.

An earlier set of 39 Wikipedia sentences with many names (for example "Aldrin") gave higher errors for every engine
(CPU-only 10.7 %, GPU-only 11.3 %, hybrid 13.2 %) with the same ranking; names the voice mispronounces count as errors for all of them.

## What a user waits for, end to end

**Measured 2026-11-23 by replay.** 62 recorded phrases (4,500 to 18,500 ms of speech each) were replayed, followed by 2,000 ms of silence, through the
live voice-detector settings in 80 ms chunks (an early detector with 250 ms of quiet and the normal one with the phrase-end quiet). Engine times are
real runs; 20 ms was added for punctuation and 10 ms for typing (both measured separately). The zero point is where the voice detector says the speech ended.
Time until the text would be typed, **mean / slowest 1 in 10** (90th percentile), in ms:

| Phrase-end quiet | CPU (language model, 4 threads) | Hybrid (live rule) | GPU (QPU kernels, greedy) |
|---|---|---|---|
| 600 ms (default) | **718** / 696 | 733 / 864 | 745 / 891 |
| 500 ms | **590** / 616 | 628 / 663 | 650 / 738 |
| 400 ms | **480** / 567 | 551 / 643 | 570 / 655 |

- Engine time alone (mean): CPU 171 ms, hybrid 261 ms, GPU 293 ms. The text is ready at about 520 / 610 / 640 ms with the default quiet, then waits for the quiet rule.
  The engine only shows up in the total when it takes longer than about 350 ms (phrases of roughly 10,000 ms or more on the CPU).
- A shorter quiet rule can only go as low as 250 ms plus the engine time; with the CPU engine that floor is about 420 ms.
- The early detector's text was used on 57 of 62 phrases at 600 ms and on all 62 at 400 ms; otherwise the engine runs again after the phrase end.
- Replacing 80 ms mic chunks by 32 ms chunks saved about 15 ms (703 against 718 ms at 600 ms) in the replay; not tried on a live microphone.

**Not included above: the stop-word check.** After the phrase-end decision the live code runs a small wake-word model on phrases of 3,000 ms or
less. Measured on 30 short recordings (mean 2,270 ms): 123 ms on the GPU (the version used) and 433 ms on the CPU. Half of the 1,054 saved phrases of the
speaker are 3,000 ms or shorter (median 2,984 ms; 86 % are 4,500 ms or shorter), so for them the typed time is about 120 ms later than the table says.
Starting the check right after the early text (setting `OIDO_KWS_EARLY=1` in `oido-type.py`) hides it; estimated, not replayed: a phrase of 3,000 ms or less is typed after about
760 ms by default, about 640 ms with the early check, about 545 ms with the early check and a 500 ms quiet rule.

**The clips above are all 4,500 ms or longer, about 14 % of the saved phrases.** For the shorter majority the engine takes about 80 ms, so the wait is almost entirely
the quiet rule plus the stop-word check.

### Shorter quiet rule: how often a phrase is split, and word errors

**Measured 2026-11-23, offline.** The same voice detector (live settings) run on 1,034 recorded phrases of 1,000 ms or longer, each of which the 600 ms rule kept in one piece:

| Phrase-end quiet | Split into 2 or more phrases | Word errors on 25 hand-labelled phrases (364 words) |
|---|---|---|
| 600 ms | 12 (1.2 %) | 28 (7.7 %) |
| 500 ms | 15 (1.5 %) | 29 (8.0 %) |
| 400 ms | 28 (2.7 %) | 32 (8.8 %) |

The word errors use the CPU engine without language model and the voice-detector cut with its padding, which by itself costs about 6 words against transcribing the whole clip (22 errors, 6.0 %).
Offline voice detector and a small sample: the real test is live use. `OIDO_SILENCE_MS` sets the quiet (default 600).

## Real recorded phrases: speed and accuracy

One speaker, phrases of 4,500 to 18,500 ms, no transcripts published. **Measured 2026-11-23, single run.**

**Speed** (66 phrases): mean wait after the audio ends, ms:

| | CPU (3 threads) | Streaming head | Live hybrid | GPU |
|---|---|---|---|---|
| All 66 phrases | **189** | 180 | 275 | 326 |
| 8 longest (8,000 to 19,000 ms) | 339 | 332 | 348 | 538 |

"Streaming head" means starting the GPU on the first part of a phrase while the speaker is still talking, cutting at a pause. It found a usable pause (at least 300 ms of
quiet and a 3,000 ms chunk) in only 2 of 66 phrases, because the dictation already splits speech at pauses and the phrases are short; it then just runs as CPU-only.
Pauses in the synthetic test clips were 400 ms of digital silence, so the 149 ms it reached there does not carry over to real speech.

**Accuracy against the speaker's own text** (25 hand-labelled phrases, 364 words, case and punctuation ignored):

| Engine | Word errors | Error rate | Mean wait |
|---|---|---|---|
| CPU only | 22 | **6.0 %** | 189 ms |
| Streaming head | 22 | 6.0 % | 177 ms |
| GPU only | 25 | 6.9 % | 315 ms |
| Live hybrid | 31 | 8.5 % | 268 ms |

The labels were made by reading the CPU engine's text and correcting it (21 of the 25 phrases were accepted unchanged), which favours the CPU engine. The 4 corrected phrases hold almost all errors
for every engine (about 46 to 54 % of their words). Differences between CPU, GPU and streaming head are within noise; the extra 9 hybrid errors match the synthetic tests.

**Language model.** The menu's CPU entry runs with the language model; on these labels it scores 36 errors (9.9 %) against 22 without, but the labels came from the no-model text, so this comparison is not fair and is unresolved.
**Streaming model.** The streaming model (`oido_stream.tnm`) scored 41 errors (11.3 %) against 22 (6.0 %) for the normal model on the same labels, with the same whole-clip time (163 against 158 ms). It would only help by processing while the speaker talks.

## Speed against clip length

**Measured 2026-11-22, median of 5 runs, ms.** Slices of the 11,000 ms `samples/jfk.wav`. "Hybrid" here is the simple version: the last 40 % of
the clip goes to the CPU, the first 60 % to the GPU, no attempt to find a pause.

| Clip | CPU only (3 threads) | GPU only | Hybrid, fixed 60 % cut |
|---|---|---|---|
| 2,500 ms | **71** | 134 | 145 |
| 5,000 ms | **158** | 245 | 210 |
| 10,000 ms | **323** | 463 | 347 |
| 15,000 ms | 531 | 782 | **465** |

Speed of each engine on its own (**2026-11-22**): CPU about 33 ms per 1,000 ms of audio; GPU 103 ms for a 1,000 ms clip, 225 ms for 4,000 ms,
363 ms for 7,000 ms, 470 ms for 10,000 ms, 746 ms for 13,000 ms, 984 ms for 17,500 ms (a staircase at 2,560 ms blocks). That boot's GPU was about 30 %
slower than the curve measured on 2026-11-20.

## Hybrid cut rules: speed and word errors

The hybrid cuts a phrase at one point, transcribes the head on the GPU and the tail on the CPU at the same time, and joins the texts.
Where to cut is a pluggable rule (`OIDO_HYBRID_RULE` in `oido-type.py` in the `oido-streaming` repo). **Measured 2026-11-23, single run, 94 clips.**

| Cut rule | Word error rate | Mean time | 4,500 to 9,000 ms clips (error / time) | 9,000 to 19,000 ms clips (error / time) |
|---|---|---|---|---|
| CPU only (no cut) | 6.6 % | 312 ms | 6.3 % / 202 ms | 6.7 % / 405 ms |
| GPU only (no cut) | 6.8 % | 531 ms | 6.7 % / 358 ms | 6.8 % / 676 ms |
| `live`: quiet moments between 28 % and 70 % of the phrase | 8.3 % | 265 ms | 9.4 % / 210 ms | 7.8 % / 311 ms |
| `balance`: balance point of the speed curves, anywhere | 9.9 % | 257 ms | 10.9 % / 197 ms | 9.4 % / 308 ms |
| `snap`: balance point moved to the quietest moment within 500 ms | 8.3 % | 256 ms | 9.2 % / 193 ms | 7.9 % / 310 ms |
| balance on fresh curves, quiet moments only | 8.1 % | 259 ms | 9.4 % / 205 ms | 7.5 % / 303 ms |

Cutting in the middle of speech (`balance`) is the fastest on long clips and the least accurate. Snapping to a quiet moment recovers most of the loss.
No hybrid rule matched CPU-only accuracy.

## Four ideas for a faster hybrid

**Measured 2026-11-23, same 94 clips, single run.** Baseline: live rule 8.3 % / 265 ms.

| Idea | Word error rate | Mean time | Result |
|---|---|---|---|
| 1. Overlap each side of the cut by 1,000 ms and stitch the two texts | 8.3 % | 301 ms | No gain. Short clips improved (9.4 to 8.3 %), long clips got worse (7.8 to 8.3 %), and it is about 36 ms slower. Stitching the per-frame outputs instead of the text is untested (the CLI prints only text). |
| 2. Start transcribing the head while the speaker is still talking | 7.5 % | **149 ms** after the speech ends | Real-time test on 47 clips with 400 ms digital-silence pauses (an earlier simulation said 191 ms). **Does not carry over to real speech** (see above: a pause was found in 2 of 66 real phrases). |
| 3. Balance the cut on freshly measured speed curves, quiet moments only | 8.1 % | 259 ms | Within noise of the live rule. |
| 4. Phrases under 9,000 ms go to CPU-only, longer ones to the hybrid | **7.3 %** | 261 ms | A point better than the live rule at the same speed. (Under 9,000 ms to GPU-only instead: 7.4 % / 333 ms.) On real phrases the hybrid gave no gain even above 8,000 ms (348 against 339 ms for CPU). |

## Other speed tests

**Measured 2026-11-23** on 25 hand-labelled phrases (364 words), CPU engine, median of 3 passes per phrase:

| Variant | Word errors | Mean wait |
|---|---|---|
| 3 threads, cores 1 to 3, low priority | 22 (6.0 %) | 180 ms |
| 3 threads, normal priority | 22 | 175 ms |
| 2 threads | 22 | 233 ms |
| **4 threads, cores 0 to 3** | **22** | **159 ms** |
| audio sped up 1.1x (time-stretch) | 25 (6.9 %) | 161 ms |
| audio sped up 1.25x | 43 (11.8 %) | 135 ms |
| audio sped up 1.5x | 56 (15.4 %) | 113 ms |

CPU time is proportional to audio length (about 30 ms per 1,000 ms, almost no fixed cost), so only less computing helps. Four threads are 12 % faster with the same words;
speeding up the audio loses words quickly. The first call for a new clip length costs only about 4 ms more on the CPU and 8 ms more on the GPU than a repeat.

## Earlier tuned hybrid numbers

**Earlier (2026-11-20), not re-run.** The tuned hybrid on single clips, before the GPU got slower on the later boot: 5,000 ms clip cut at 2,500 ms
141 ms (GPU only 238 ms); 10,000 ms clip 243 ms (463 ms); 11,000 ms `jfk.wav` 267 ms (582 ms); 15,000 ms clip cut at 5,100 ms 313 ms (781 ms). On 82 recorded
phrases the median went from 323 ms (GPU only) to 214 ms. These were not scored for word errors against known text, and the real-phrase test above found no gain over CPU-only.

The resident CPU engine took about 32.5 ms per 1,000 ms of audio plus 1 ms on 3 threads (exact build). The 4x4 NEON matmul, 4-row attention and
cached position terms made the CPU engine about 30 % faster than the earlier build (10,000 ms clip about 570 to 400 ms in the one-shot CLI).
An earlier claim of 1.8x came from a build whose results were not bit-exact and was withdrawn.

## Dot-product (sdot) builds

**Earlier (2026-11-20), not re-run, Pi 5 untested.** The Pi 400's Cortex-A72 has no dot-product instructions, so these builds are for phones and the Pi 5.
oido with 4x4 sdot matmul and 4-row sdot attention, logits bit-exact against the plain build, on a Pixel 9 (3 A720 cores): 10,000 ms clip
94.7 to 62.3 ms (1.49 to 1.56x); one A720 core 1.70 to 1.72x; one-shot CLI 1.20 to 1.55x. On the Pi 400 the same source gave 321 to 315 ms.

## heart and heart-nano

**Earlier (2026-11-17 to 2026-11-21), not re-run for this page**, text-to-speech engines (sanoTTS) running on the Pi 400 unless noted.

| Change | Before | After | Note |
|---|---|---|---|
| NEON decoder, heart-nano, 7,000 ms of audio, one-shot CLI | 104 to 123 ms | 48 to 72 ms (decoder 72 to 82 -> 20 ms) | at most 1 LSB different |
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
- **Speeding up the audio before transcribing**: faster, but words are lost (see [other speed tests](#other-speed-tests)).
- **The streaming model as a replacement**: 11.3 % word errors against 6.0 % on the labelled phrases.
- **heart-nano int8 sdot**: slower on the Pixel 9 (decoder 5.5 to 6 ms) and 27 dB SNR.
- **Converting heart-nano int8 weights to f32**: 0.72 to 0.77x (slower).
- **Hand-scheduled NEON assembly, a lock-free thread pool, a 64-row GPU kernel, smaller workgroups and several other GPU experiments**: no gain or slower; the
  GPU details are in [gpu-port-kit](https://github.com/tarelerulz/gpu-port-kit).

## Limits

- **Synthetic voice for the large accuracy set.** The 94-clip word error numbers use the heart TTS voice. Real voices have higher errors for every engine,
  and the gaps may change. The real-phrase accuracy rests on only 25 labelled phrases of one speaker, labelled by correcting the CPU engine's text, which favours it.
- **Single runs.** No repeats of the accuracy runs and no significance test.
- **Replays, not live use.** The end-to-end and quiet-rule numbers replay recordings with silence added; nothing was measured on a live microphone, and the stop-word check is not in the end-to-end table.
- **Most phrases are shorter than the benchmark clips.** The benchmark clips are 4,500 ms or longer, about 14 % of the saved phrases.
- **No listening checks** of the hybrid or sdot output.
- **Pi 400 only for the oido numbers**; the Pi 5 and phone are untested or earlier-only.
- **No language model** in the large word-error runs; with it the errors would be lower on some phrases and the comparison on the labelled phrases is unresolved.
- Clock and governor: performance governor; the GPU (V3D) clock was read back as 800 MHz (`v3d_freq=800`, `over_voltage=6` in the boot config) after these runs. The CPU was not overclocked (1,800 MHz).

## Reproduce

Build the CPU engine with the resident mode used by the hybrid, then feed it WAV paths on stdin; it prints `path<TAB>text` per line:

```sh
cd host && make tasr_cli_a72
printf '../samples/jfk.wav\n' | taskset -c 1,2,3 nice -n 19 ./tasr_cli_a72 nemo8.tnm --serve
```

The `--serve` mode was added to this repo together with this page. The settings mentioned above (`OIDO_SILENCE_MS`, `OIDO_KWS_EARLY`, `OIDO_HYBRID_RULE`)
are in `oido-type.py` in the [oido-streaming](https://github.com/tarelerulz/oido-streaming) repo.

The benchmark scripts are in [`bench/`](bench/README.md): speed against clip length, the cut rules, the four ideas, the end-to-end replay, the quiet-rule test and the speed lab, plus
the everyday test set (sentences and a script that builds the audio) and the output of the everyday-set runs in `bench/results/`. The recordings and the hand-written labels behind the
real-phrase numbers are one person's voice and are not published: the scripts take your own recordings (`CLIPS_DIR`) and labels (`label_clips.py` makes them). The stop-word check timing is not included.

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

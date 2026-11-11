# Benchmarks: wall-clock speed of this build vs upstream oido v2

All numbers are measured on the hardware named below. Nothing here is a word-error benchmark.

## Test setup

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

## Pi 400 (Cortex-A72, NEON only, no DOTPROD)

Cores pinned with `taskset`. Build with `-O3 -march=armv8-a -mtune=cortex-a72 -DTASR_NW=8`.

| Clip (length) | Cores | This build | v2a | v2b |
|---|---|---|---|---|
| tarel40 (10.00 s) | 0–2 | **529 ms** (504–683) | 929 ms (915–951) | 757 ms (743–768) |
| movie10 (10.00 s) | 0–2 | **569 ms** (508–631) | 957 ms (945–975) | 775 ms (767–801) |
| en16 (3.85 s) | 0–2 | **216 ms** (202–235) | 374 ms (369–385) | 316 ms (311–334) |
| tarel40 (10.00 s) | 0–3, 4 threads | **527 ms** (486–663) | 925 ms (921–961) | 752 ms (745–771) |

On the Pi, this build is the fastest in wall-clock time on every clip. v2 uses less CPU time (about 1.9× less for v2a and 2.4× less for v2b, measured per process), but it does not finish sooner.

## Android phone (Pixel 9 Linux Terminal VM, ARMv8.2 + DOTPROD)

Pinned to the four fast cores (4–7), 4 threads, `TASR_THREADS=4`. Build: `-march=armv8.2-a+dotprod`.

| Clip (length) | This build | v2a | v2b |
|---|---|---|---|
| tarel40 (10.00 s) | **141 ms** (137–158) | 166 ms (129–185) | 150 ms (142–164) |

On the phone's fast cores this build is faster than both v2 models: about 6% faster than v2b and about 15% faster than v2a. The v2b gap is small, and the run ranges overlap.

### DOTPROD and NEON-only (this build, same phone, same clip)

| Build | Median (min–max) |
|---|---|
| DOTPROD (`-march=armv8.2-a+dotprod`) | 130 ms (116–273) |
| NEON only (`-march=armv8-a`) | 161 ms (149–491) |

DOTPROD gives about 20% here. Upstream nemo8 gains more from DOTPROD (about 2.8×) because its build lacks the NEON changes in this repo.

## Things to know before comparing

- **The two engines are not the same code.** This build has the NEON and DOTPROD changes from `optimization_log.md`. The v2 numbers use the v2.0.0 tree as released, built with the same compiler flags. A same-source comparison would be needed to attribute the gap to specific changes. The upstream nemo8 build was measured to be slower than this build on the same phone (see below), so the gap is not only the model.
- **Upstream nemo8 vs this build (phone, big cores, 4 threads, tarel40):** upstream sources 260 ms median; this build 141 ms. The upstream numbers were built from the v2.0.0 source tree, not from this repo.
- **The model is not the same as v2.** v2b makes more word errors than this build's nemo8 on `movie10.wav`. No word-error rate was computed on these clips; they are too few to be a benchmark.
- **The licences differ.** This repo's code is GPL-3. The v2 weights are CC BY-NC-SA 4.0 (non-commercial). This build uses the upstream nemo8 weights.
- **Upstream phone RTFx numbers** in `docs/phone-benchmarks.md` (for example 121× on four threads) were measured on a different phone and on a different clip. They don't match the 4-fast-core timing above (about 38× for the same model on the Pixel 9 VM). The cause is not established.

## Reproduce

```sh
cd esp32/host            # or host/ in this repo
make tasr_cli_a72        # Pi 400
# or
make tasr_cli_phone      # Android, ARMv8.2 + DOTPROD

taskset -c 0-2 ./tasr_cli_a72 nemo8.tnm --lm nemo_lm.tlm --beam 4 tarel40.wav
```

Time each run with `date +%s%N` before and after the command. Use 10 runs per model and report the median.

## Raw data

The raw per-run timings and the scripts used are kept with the author's private test notes, not in this repo.

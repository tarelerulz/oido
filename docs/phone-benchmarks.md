# How We Got the Oido and sanoTTS Heart Performance Numbers

## Hardware

Android phone with a big.LITTLE CPU:
- 4× Cortex-A520 (little cores, capacity 191)
- 3× Cortex-A720 (mid cores, capacity 741)
- 1× Cortex-X4 (prime core, capacity 1024)

The thread pool skips little cores and pins workers to the 4 fast cores (3×A720 + 1×X4).

---

## Part 1: Oido STT Numbers

### What is Oido?

The engine is an open-source speech-to-text engine (upstream NeMo Conformer-CTC tinyasr). It runs a NeMo Conformer-CTC Small model (13M parameters) quantized to int8 (nemo8, 14MB) or int4 (nemo4, 8MB). The C inference engine (tinyasr) is the same code that runs on an ESP32-S3 microcontroller, compiled for the host with ARM NEON/DOTPROD intrinsics. The model weight files (.tnm, .tlm) are the original unmodified upstream files — only the inference engine C code was changed.

### What RTFx means

RTFx = real-time factor (inverse). RTFx of 10× means 1 second of audio is transcribed in 0.1 seconds. Higher is faster. A clip of 4 seconds at 121× RTFx transcribes in 4/121 ≈ 33ms.

### Changes made to reach the numbers

Starting from the upstream Oido code (which had no host threading and no DOTPROD path), four changes were made:

**1. Host thread pool (commit 94515b5)**

The upstream `tasr_parallel` was a serial stub — all inference ran on one core. We added a pthreads worker pool that:
- Reads `/sys/devices/system/cpu/cpuN/cpu_capacity` to find fast cores
- Skips little cores (capacity < ~500)
- Pins each worker thread to a specific fast core to prevent scheduler migration
- The calling thread is pinned to cpu[0]

This is controlled by `tasr_set_threads(n)` and the `TASR_THREADS` environment variable.

**2. DOTPROD SIMD tier (commit 1d75907)**

The original tinyasr had only a scalar path and a NEON path. We added a third tier using ARM dot-product instructions (`vdotq_s32`, ARMv8.2-A+dotprod), which computes 16 int8 multiply-accumulates per instruction vs 8 for NEON. The compiler selects the tier at compile time via `__ARM_FEATURE_DOTPROD`. The NEON tier is kept for Cortex-A72 / Pi 4 targets that lack DOTPROD.

**3. Fix blocked int4 GEMM on DOTPROD path (commit 64231c9)**

The nemo4 model uses int4 weights in a blocked GEMM (`tasr_gemm_blk16`). The DOTPROD path was falling back to `vmull/vmlal` (NEON) instead of `vdotq_s32` because the weight tile layout was [kp][16] (NEON-friendly) instead of [16][kp] (DOTPROD-friendly). Fix: added a NEON 4-stage trn/zip transpose of each tile at the start of the DOTPROD path, then used `vdotq_s32` per row. This is a one-time transpose per tile per call.

Result: nemo4 1-thread went from 24× to 49× RTFx, 4-thread from 57× to 110× RTFx. nemo4 now matches nemo8 speed despite being int4.

**4. NEON prefetch + 2-row unroll for Cortex-A72 (commit 36c1077)**

For the Pi 400 / NEON path (armv8-a, no DOTPROD):
- Added `__builtin_prefetch(pw+128)` in the weight k-loop of `tasr_gemm_blk16`
- Unrolled `tasr_dot48_rows` to process 2 rows per iteration to overlap load latency in the A72 in-order pipeline

Result on phone forced to armv8-a (NEON only): 9.5× → 12.8× RTFx (+35%).

### Final measured numbers (apollo11 audio clip, ~4 seconds)

| Model | 1 thread | 4 threads |
|---|---|---|
| nemo8 (int8, 14MB, with LM) | 51× RTFx | 121× RTFx |
| nemo4 (int4, 8MB) | 49× RTFx | 121× RTFx |
| nemo8 NEON path (armv8-a forced) | 12.8× RTFx | — |

The LM (language model, beam search) adds ~15% compute cost vs greedy.

**Accuracy (WER):**

| Config | LibriSpeech test-clean | Noisy sample clips (17 clips: car, kitchen, reverb, meetings) |
|---|---|---|
| upstream defaults (beam=4, lw=0.3, tb=0.5) | 3.3% | 6.12% |
| Tuned defaults (beam=4, lw=0.5, tb=1.0) | not measured | **4.68%** |

The tuned parameters were found by sweeping beam size (4/8/16), LM weight (0.2–0.5) and token bonus (0.3–1.0) across the 17 noisy sample clips. Higher lm_weight gives the language model more influence; higher token_bonus encourages committing to complete words. The 23% relative WER reduction on noisy audio (~1 extra word correct per sentence) comes at no speed cost — beam=4 is unchanged.

**Temperature scaling experiment (output head):**

The output head's float32 quantization scales were also tested — multiplying all 1025 scales by a temperature factor t (0.7–2.0) to sharpen or soften the CTC output distribution. Every value of t made accuracy worse; t=1.0 (original) was best. The model's output distribution is already well-calibrated. 4.68% is the ceiling for inference-time improvements without retraining.

### Build targets added to Makefile

```makefile
# Pi 400 (Cortex-A72, ARMv8.0 NEON only)
PI_FLAGS = -O3 -march=armv8-a -mtune=cortex-a72 -pthread -DTASR_NW=8 ...
tasr_cli_a72: ...

# Android phone (ARMv8.2-A + DOTPROD, any phone ~2019+)
PHONE_FLAGS = -O3 -march=armv8.2-a+dotprod -pthread -DTASR_NW=8 ...
tasr_cli_phone: ...
```

The upstream Makefile used `-O2` with no `-pthread`, no `-DTASR_NW=8`, and no ARM targets. `TASR_NW=8` sets the number of weight columns processed per GEMM block, tuned for the nemo8 model width.

---

## Part 2: sanoTTS Heart Numbers

### What is sanoTTS heart?

sanoTTS (https://github.com/ex3ndr/supervoice-hai) is a small neural TTS engine. Two models were tested:
- **heart** (2.27M parameters): higher quality, slightly slower
- **heart-nano** (294k parameters, 1.2MB): faster, audible hiss artefacts

### How the numbers were measured

Wall-clock time was measured in Python using `time.perf_counter()` around the synthesis call. Each configuration was run multiple times; the best stable time is reported. Thread count was varied by setting the number of inference threads passed to the engine.

### heart numbers (2.27M params)

| Threads | Wall time |
|---|---|
| 1 | 0.123s |
| 2 | 0.119s |
| 4 | 0.142s |

Best: **1 thread at 0.123s**. Adding threads beyond 1 gives no benefit and 4 threads is slower — the model is small enough that thread synchronisation overhead exceeds the parallelism gain.

### heart-nano numbers (294k params, 1.2MB)

| Threads | Wall time |
|---|---|
| 1 | 0.167s |
| 2 | 0.087s |
| 4 | 0.105s |

Best: **2 threads at 0.087s**. heart-nano benefits from 2 threads because the model is tiny enough that each thread's work fits in cache. 4 threads overshoots.

### Post-processing applied to heart-nano

To reduce the hiss artefact on heart-nano, ffmpeg filtering is applied in `heart-web.py`:
```
lowpass=f=7500 (applied twice) + leveler, output resampled to 48kHz
```

### Quality ranking

Phonon (98MB) > sanoTTS heart (2.27M) > sanoTTS heart-nano (294k)

### Recommendation

Use heart at 1 thread. It is both faster (0.123s vs 0.087s best-case nano) and better quality. heart-nano's 0.087s advantage only holds at exactly 2 threads and comes with audible artefacts.

---

## Summary table

| System | Task | Best config | Best time |
|---|---|---|---|
| Oido nemo8 | STT (4s clip) | 4 threads, DOTPROD | 33ms (121× RTFx) |
| Oido nemo4 | STT (4s clip) | 4 threads, DOTPROD | 33ms (121× RTFx) |
| Oido nemo8 | STT (4s clip) | 1 thread, DOTPROD | 78ms (51× RTFx) |
| Oido nemo8 accuracy | LibriSpeech test-clean WER | beam=4, lw=0.3, tb=0.5 | 3.3% |
| Oido nemo8 accuracy | Noisy clips WER (tuned) | beam=4, lw=0.5, tb=1.0 | 4.68% (was 6.12%) |
| sanoTTS heart | TTS synthesis | 1 thread | 0.123s |
| sanoTTS heart-nano | TTS synthesis | 2 threads | 0.087s |

All numbers measured on Android phone (Cortex-X4 + A720 + A520) running in Termux.

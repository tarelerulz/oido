# Oido Optimization Log

## 1. Host Thread Pool for Multi-Core Inference (Android/Linux)
- Added pthread worker pool replacing serial tasr_parallel stub
- tasr_set_threads(n): auto-selects fast cores via /sys/devices/system/cpu/cpuN/cpu_capacity, skipping little cores
- Workers pinned to fast cores; calling thread pinned to cpu[0] to prevent scheduler migration
- TASR_THREADS env var overrides (1=serial, 0/unset=auto)
- Result: 4-thread speedup ~2.2x no-LM, ~2.3x with LM

## 2. DOTPROD (sdot) SIMD Tier for aarch64
- Added TASR_DOTPROD tier using vdotq_s32 (ARMv8.2 dot-product extension)
- NEON tier kept for Cortex-A72 / Pi 4 targets
- Note: Pi 400 (Cortex-A72) does NOT have DOTPROD — uses NEON path

## 3. Fix tasr_gemm_blk16 DOTPROD Path
- Blocked int4 GEMM (nemo4 model) was using vmull/vmlal instead of vdotq_s32
- Fix: NEON 4-stage trn/zip transpose of tile [kp][16]->[16][kp] + vdotq_s32 per row
- Result: nemo4 1-thread 24x -> 49x RTFx; 4-thread 57x -> 110x RTFx (on phone)
- nemo4 now matches nemo8 speed

## 4. NEON Prefetch + 2-Row Unroll for Cortex-A72 (Pi 400)
- tasr_gemm_blk16 NEON: __builtin_prefetch(pw+128) in k-loop
- tasr_dot48_rows NEON: 2-row unroll to overlap loads in A72 pipeline
- Result on the Pi 400 (wall clock, 10 s clip, medians, transcripts identical): about 13% faster at 1 thread (1.646 s -> 1.438 s) and about 5-7% faster at 4 threads (0.633 s -> 0.589 s).
- The earlier "+35%" figure was measured on a phone with the NEON path forced (`-march=armv8-a`), not on the Pi. It is not the Pi result; see BENCHMARKS.md.
- Tried and rejected on the Pi (all within about 5% noise or worse): `-mcpu=cortex-a72`, `-O2` instead of `-O3` (about 10% slower), prefetch in the x-stationary GEMM, 2-row unroll in attention.

## Measured RTFx (apollo11 audio, 4-second clip, phone)
- nemo8 (int8, 14MB): 1t=51x, 4t=121x
- nemo4 (int4, 8MB): 1t=49x, 4t=121x

## Measured RTFx NEON path (2-second tone, phone forced armv8-a)
- nemo8 old: 9.5x, new: 12.8x (+35%)

## Files Transferred to Pi 400
- nemo4.tnm copied to ~/oido/models/ (2026-10-31)

## Next: Ideas to Test on Pi 400 (target 95-100x RTFx)
See speedup_ideas.md

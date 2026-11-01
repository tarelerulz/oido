# Oido Pi 400 Speedup Ideas (target: 95-100x RTFx)

1. **LUT-based int4 GEMM** — precompute 4-bit product lookup tables, replacing multiply with table lookups on A72

2. **Switch to nemo4 model** — if Pi is still running nemo8, int4 halves the compute

3. **4-thread build** — Pi 400 has 4 cores; thread pool may not be enabled there yet

4. **-mcpu=cortex-a72 tuning flag** — A72-specific instruction scheduling by compiler

5. **Cache-blocked GEMM** — tile sizes matched to A72 32KB L1 / 1MB L2

6. **Disable language model** — no-LM mode already measurably faster

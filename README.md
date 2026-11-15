# oido: offline speech-to-text for Android (Termux) and Raspberry Pi 400

Build the C speech engine on a phone (Termux) or on a Pi 400, and use it from nano:
Ctrl+T records a few seconds and replaces `[oido]` with the transcript.

## Speed at a glance

Wall-clock time to transcribe a 10-second clip, median of 10 runs, beam 4 with the language model (full table and method in [BENCHMARKS.md](BENCHMARKS.md)):

| Hardware | This build | Upstream oido v2a | Upstream oido v2b |
|---|---|---|---|
| Raspberry Pi 400, 3 cores | 529 ms | 929 ms | 757 ms |
| Android phone (Pixel 9, Linux Terminal VM), 4 fast cores | 141 ms | 166 ms | 150 ms |

On the Pi this build is the fastest in wall-clock time. v2 uses less CPU time but finishes later. Only one clip is shown; these are not word-error benchmarks.

**GPU:** the encoder also runs on the Pi 400's GPU, level with the CPU on a 10 s clip (0.55 s vs 0.58 s wall) with the CPU cores almost idle. See [Running on the Pi 400 GPU](#running-on-the-pi-400-gpu-experimental) and the [full write-up](https://github.com/tarelerulz/gpu-port-kit/blob/main/OIDO_GPU_STORY.md).

## Credits

This is a fork of **Oído** by Lokutor ([lokutor-ai/oido](https://github.com/lokutor-ai/oido), GPL-3.0). The speech engine, models, and original documentation are Lokutor's work. Changes in this repo, including the Raspberry Pi 400 and Android (Termux) builds, the nano dictation hook, and the phone scripts, are by tarel.

Model weights are from Lokutor's Hugging Face pages:
- [oido-ctc-small-int8](https://huggingface.co/lokutor-ai/oido-ctc-small-int8) (CC-BY-4.0)
- [oido-ctc-small-int4](https://huggingface.co/lokutor-ai/oido-ctc-small-int4) (CC-BY-SA-4.0)
- [oido-ctc-small-stream-int8](https://huggingface.co/lokutor-ai/oido-ctc-small-stream-int8) (CC-BY-SA-4.0)
- [oido-es-ctc-small-int8](https://huggingface.co/lokutor-ai/oido-es-ctc-small-int8) (CC-BY-4.0)

## Contents

- `components/tinyasr/`: the C engine (NeMo Conformer-CTC, language model, segmenter).
  Includes the host thread pool (`tasr_set_threads`, skips little cores) and the DOTPROD path.
- `host/Makefile`: build targets (`tasr_cli_phone`, `tasr_cli_a72`, `bench`).
- `host/tasr_cli.c`: command-line front end.
- `host/bench.c`: timing benchmark on synthetic clips.
- `host/pi400_setup.sh`: builds `tasr_cli_a72` on a Pi 400 and installs `oido_nano.sh`.
- `oido_nano.sh`: Pi 400 nano hook (ALSA `arecord`).
- `phone/oido_nano.sh`: phone (Termux) nano hook (`parecord`, optional punctuation).
- `phone/oido_web.py`: small web server with punctuation.
- `phone/transcribe.py`: batch transcription of 16 kHz WAV files.
- `phone/transcribe_apollo.sh`: long-file test script.
- `port/gpu/`: GPU (Vulkan, Pi 400) work; the story and the code are in [gpu-port-kit](https://github.com/tarelerulz/gpu-port-kit) ([OIDO_GPU_STORY.md](https://github.com/tarelerulz/gpu-port-kit/blob/main/OIDO_GPU_STORY.md)).
- `BENCHMARKS.md`: timing method, Pi and phone results, and comparison with upstream v2.
- `docs/phone-benchmarks.md`: measured numbers and the changes behind them (Part 2, sanoTTS heart, is not part of this engine).

## Model files (not included)

The CLI needs a `.tnm` model and optionally a `.tlm` language model. Put them where the scripts expect:

- phone: `~/tasr_bench/nemo8.tnm` and `/sdcard/oido/models/nemo_lm.tlm`
- Pi 400: `~/oido/models/nemo8.tnm` and `~/oido/models/nemo_lm.tlm`

The models are on Hugging Face ([int8](https://huggingface.co/lokutor-ai/oido-ctc-small-int8), [int4](https://huggingface.co/lokutor-ai/oido-ctc-small-int4), [streaming](https://huggingface.co/lokutor-ai/oido-ctc-small-stream-int8)). To check a download, compare its SHA-256 (`sha256sum file`):

| file | SHA-256 |
|---|---|
| `nemo8.tnm` | `452e332e21f46f161e383a1f26f36294785b2782cb3252928841dd7e6480c68b` |
| `nemo_lm.tlm` | `00f174f83ad70d865301304fd51897839c542d6f887f8e93e4e1b83938566a1f` |
| `nemo4.tnm` | `e6fad1c3a1c0065add9fe3c9f10d6fa075b09b27c11f1a830b35f31fabf7d5e9` |
| `oido_stream.tnm` | `d815ad97ab3def101a267531db0778395c7e8eab9c09571ae4cabca71f5faa11` |

These are the hashes Hugging Face lists for those files; the author's own copies match them.

Paths are set at the top of each script.

## Build the phone version (Termux, aarch64, ARMv8.2 + dotprod)

    pkg install clang make python
    cd host
    make tasr_cli_phone   # builds host/tasr_cli_phone (prebuilt copies are in phone/bin)
    mkdir -p ~/tasr_bench && cp tasr_cli_phone ~/tasr_bench/tasr_cli_1t
    ./tasr_cli_phone /path/to/nemo8.tnm --lm /path/to/nemo_lm.tlm /path/to/audio16k.wav

Audio must be 16 kHz mono 16-bit WAV. Multi-threaded: `TASR_THREADS=4` (default: fast cores).

Optional punctuation (used by `phone/oido_nano.sh` and `phone/oido_web.py` if present):
a sherpa-onnx punctuation CLI at `~/punct_cli` and its model. Without them the raw text is used.

`phone/oido_web.py` listens on `127.0.0.1:5000` (this device only), sends no CORS header, and refuses POSTs from other websites or with a foreign `Host`. Set `OIDO_WEB_HOST=0.0.0.0` to reach it from other devices.

## What you should see

Try it on the public sample `samples/jfk.wav` (11 s, 16 kHz mono; the first sentence of a 1961 speech by US President Kennedy, from the whisper.cpp repository, MIT). With `nemo8.tnm` and `nemo_lm.tlm` (see [Model files](#model-files-not-included) for the SHA-256 values) the output is:

    $ ./tasr_cli_a72 nemo8.tnm --lm nemo_lm.tlm --beam 4 samples/jfk.wav
    samples/jfk.wav	and so my fellow americans ask not what your country can do for you ask what you can do for your country

Wall-clock time for that command, median of 10 runs (min to max), measured 2026-11-14:

| Device | Build | Time | Speed vs real time |
|---|---|---|---|
| Raspberry Pi 400, 3 cores | `tasr_cli_a72` (NEON) | 597 ms (561–636) | about 18× |
| Pixel 9, Linux Terminal VM, 4 threads on cores 4 to 7 | `tasr_cli_phone` (dotprod) | 130 ms (118–336) | about 85× |
| Pixel 9, Linux Terminal VM, 4 threads on cores 4 to 7 | NEON only (`-march=armv8-a`) | 170 ms (160–306) | about 65× |
| Pixel 9, Termux, 2 threads | `tasr_cli_phone` (dotprod) | 398 ms (271–473) | about 28× |
| Pixel 9, Termux, 2 threads | NEON only (`-march=armv8-a`) | 495 ms (338–580) | about 22× |

Dotprod was faster than NEON-only at every thread count tried, by about 10 to 30%. The Pixel 9 numbers were measured over SSH, where Android let Termux use cores 0 to 6 but not the fast core 7. With the default four threads this build then took about 1.0 s, because four threads share only three fast cores. If the phone build is slow for you, try `TASR_THREADS=2` or `TASR_THREADS=3`. The Linux Terminal VM rows (all four fast cores available, pinned with `taskset -c 4-7`, median of 30 runs alternating the two builds) were measured on 2026-11-14 with a build of this repository; single runs varied by up to 2×, so treat them as roughly 100 to 150 ms. The 141 ms figure in the speed table above is for a different, unpublished 10 s clip; on that clip the same VM setup gave 118 ms with dotprod and 159 ms NEON-only on 2026-11-14. No Raspberry Pi 5 numbers exist yet.

## Raspberry Pi 5

Not tested on a Pi 5. The Pi 5's Cortex-A76 has the ARMv8.2 dot-product instructions, so `make tasr_cli_phone` (the phone build, which uses them) should work there. As a stand-in, that build was compiled and run on 2026-11-14 on an Android phone (Pixel 9, Termux, ARMv8.2 with dot product): it built from a clean copy of this repo in about 10 s and transcribed the 11 s JFK sample correctly, the same text as the Pi 400 build. That shows the code and Makefile work on an ARMv8.2 + dotprod CPU, not that a Pi 5 does. If you try it on a Pi 5, please open an issue with the transcript and timing. (`make tasr_cli_a72` should also work, since it only needs NEON.)

## Build for Pi 400 (Cortex-A72, NEON only)

    cd host
    make tasr_cli_a72
    bash pi400_setup.sh   # also installs ~/oido_nano.sh

## Running on the Pi 400 GPU (experimental)

The encoder also runs on the Pi 400's GPU (V3D 4.2, via Vulkan), with the CPU cores almost idle. On a 10-second clip the GPU path took 0.54 s of GPU time on 2026-11-13 (wall time 0.55 s against 0.58 s for the CPU build), down from 3.3 s for the first working version. On a 5-second clip it was 0.30 s against 0.28 s on the CPU. It gives the same text as the CPU build on the test clips, but it decodes greedily, without the language model, so unusual words can come out with stray letters. For the language model on the GPU path, build `host/tasr_lmdecode` (`make -C host tasr_lmdecode`) and see "Decoding" in the kit README. It is not faster than the CPU on this board, only level with it. It has not been voice-tested in daily dictation.

How it got there (data layout, turning every heavy step into one matrix multiply, a hand-written QPU assembly matmul, a higher GPU clock, then removing wasted work and jobs, plus what did not help) is written up in [OIDO_GPU_STORY.md](https://github.com/tarelerulz/gpu-port-kit/blob/main/OIDO_GPU_STORY.md). The code and benchmarks are in [gpu-port-kit](https://github.com/tarelerulz/gpu-port-kit).

**Trying it yourself.** You need a Raspberry Pi 4 or 400 (V3D 4.2) with Vulkan, `glslc`, Python 3 with numpy and a C compiler. In short:

1. Clone [gpu-port-kit](https://github.com/tarelerulz/gpu-port-kit) and build: `sh tools/build_exec_big.sh && sh tools/build_exec_server.sh && sh tools/build_exec_hybrid.sh && sh tools/build_speech_app_shaders.sh`.
2. Download the nemo8 weights from [Hugging Face](https://huggingface.co/lokutor-ai/oido-ctc-small-int8) and set `GPUPORT_MODEL` to the file.
3. Run the checks: `python3 tests/run_all.py` (about 2 minutes, ends with `ALL KIT CHECKS PASS`).
4. Transcribe a clip: `python3 tools/transcribe.py clip.wav` (16 kHz mono 16-bit WAV). Add `GPUPORT_ASM=1` for the fast hand-written matmul.

On 2026-11-14 these steps were run from a fresh clone with an empty home folder on the author's Pi 400: the checks pass and the 11 s JFK sample comes out right in 1.2 s of GPU time (0.7 s with `GPUPORT_ASM=1`), the same text as the CPU build. The Vulkan folders were still the author's, so a stock Raspberry Pi OS install was not tested. This is research code, not an installer; the kit README has the details and the environment variables. If something does not work, please open an issue.

## License

The code is GNU GPL v3 (see `LICENSE`, copied unchanged from [lokutor-ai/oido](https://github.com/lokutor-ai/oido)). Copyright notices in the source files are unchanged.

Model weights are not in this repository. Each one is licensed by Lokutor as listed under Credits (CC-BY-4.0 or CC-BY-SA-4.0).

## Prebuilt phone binaries (phone/bin)

Android aarch64 builds (NDK r30, API 24, ARMv8.2 + dotprod), built on the phone from the matching source on 2026-10-31 (exact match to this tree not verified):

- `phone/bin/tasr_cli_phone`: the CLI (same as `make tasr_cli_phone`)
- `phone/bin/tasr_cli_1t`: single-thread CLI used by `phone/oido_nano.sh`
- `phone/bin/tasr_cli_mt`: multi-thread CLI used by `phone/transcribe_apollo.sh`
- `phone/bin/libtinyasr-android.so`: the engine as a shared library

Rebuilding from this source with the NDK may give different bytes.

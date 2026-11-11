# oido: offline speech-to-text for Android (Termux) and Raspberry Pi 400

Build the C speech engine on a phone (Termux) or on a Pi 400, and use it from nano:
Ctrl+T records a few seconds and replaces `[oido]` with the transcript.

## Speed at a glance

Wall-clock time to transcribe a 10-second clip, median of 10 runs, beam 4 with the language model (full table and method in [BENCHMARKS.md](BENCHMARKS.md)):

| Hardware | This build | Upstream oido v2a | Upstream oido v2b |
|---|---|---|---|
| Raspberry Pi 400, 3 cores | 529 ms | 929 ms | 757 ms |
| Android phone, 4 fast cores | 141 ms | 166 ms | 150 ms |

On the Pi this build is the fastest in wall-clock time. v2 uses less CPU time but finishes later. Only one clip is shown; these are not word-error benchmarks.

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
- `BENCHMARKS.md`: timing method, Pi and phone results, and comparison with upstream v2.
- `docs/phone-benchmarks.md`: measured numbers and the changes behind them (Part 2, sanoTTS heart, is not part of this engine).

## Model files (not included)

The CLI needs a `.tnm` model and optionally a `.tlm` language model. Put them where the scripts expect:

- phone: `~/tasr_bench/nemo8.tnm` and `/sdcard/oido/models/nemo_lm.tlm`
- Pi 400: `~/oido/models/nemo8.tnm` and `~/oido/models/nemo_lm.tlm`

Paths are set at the top of each script.

## Build the phone version (Termux, aarch64, ARMv8.2 + dotprod)

    pkg install clang make python
    cd host
    make tasr_cli_phone
    mkdir -p ~/tasr_bench && cp tasr_cli_phone ~/tasr_bench/tasr_cli_1t
    ./tasr_cli_phone /path/to/nemo8.tnm --lm /path/to/nemo_lm.tlm /path/to/audio16k.wav

Audio must be 16 kHz mono 16-bit WAV. Multi-threaded: `TASR_THREADS=4` (default: fast cores).

Optional punctuation (used by `phone/oido_nano.sh` and `phone/oido_web.py` if present):
a sherpa-onnx punctuation CLI at `~/punct_cli` and its model. Without them the raw text is used.

## Build for Pi 400 (Cortex-A72, NEON only)

    cd host
    make tasr_cli_a72
    bash pi400_setup.sh   # also installs ~/oido_nano.sh

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

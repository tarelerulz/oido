# oido: offline speech-to-text for Android (Termux) and Raspberry Pi 400

Build the C speech engine on a phone (Termux) or on a Pi 400, and use it from nano
(Ctrl+T dictation on Pi 400, see `oido_nano.sh`).

## Contents

- `components/tinyasr/`: the C engine (NeMo Conformer-CTC, language model, segmenter).
- `host/Makefile`: builds the CLI for each target.
- `host/tasr_cli.c`: command-line front end (`tasr_cli <model.tnm> [--lm lm.tlm] <audio.wav>`).
- `host/pi400_setup.sh`: builds `tasr_cli_a72` on a Pi 400 and installs `oido_nano.sh`.
- `oido_nano.sh`: nano speller hook; records 5 s and replaces `[oido]` with the transcript.
- `optimization_log.md`, `speedup_ideas.md`: measurements and open speed ideas (phone and Pi).

Model weights are not included. The CLI needs a `.tnm` model and optionally a `.tlm` language model.

## Build the phone version (Termux, aarch64, ARMv8.2 + dotprod)

    pkg install clang make
    cd host
    make tasr_cli_phone
    ./tasr_cli_phone /path/to/model.tnm --lm /path/to/lm.tlm /path/to/audio16k.wav

Audio must be 16 kHz mono 16-bit WAV.

## Build for Pi 400 (Cortex-A72, NEON only)

    cd host
    make tasr_cli_a72

## License

GNU GPL v3 (see `LICENSE`). Copyright notices in the source files are unchanged.

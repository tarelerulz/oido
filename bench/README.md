# Benchmark scripts

The scripts behind the numbers in [../BENCHMARKS.md](../BENCHMARKS.md). All times in milliseconds. They were written for one Raspberry Pi 400 (4 cores:
the CPU engine runs on cores 1 to 3 or 0 to 3 and the GPU helper on core 3), so on other boards change the core lists. Run them one at a time and keep the
board otherwise idle: they time the engines, and running two at once changes the results.

Paths come from environment variables (`paths.py`); the defaults work from a clone of this repo.

| Variable | Meaning | Default |
|---|---|---|
| `OIDO_CLI`, `OIDO_MODEL`, `OIDO_LM` | the CPU engine (`make -C host tasr_cli_a72`, built with the `--serve` mode) and its model files | `host/tasr_cli_a72`, `models/nemo8.tnm`, `models/nemo_lm.tlm` |
| `GPU_PORT_KIT` | clone of [gpu-port-kit](https://github.com/tarelerulz/gpu-port-kit) (the GPU engine) | `~/gpu-port-kit` |
| `OIDO_TYPE` | `oido-type.py` from [oido-streaming](https://github.com/tarelerulz/oido-streaming) (hybrid class, cut rules) | `~/oido-streaming/oido-type.py` |
| `SILERO_VAD` | Silero voice-activity model (`silero_vad.onnx`), used through `sherpa_onnx` | `models/silero_vad.onnx` |
| `CLIPS_DIR` | **your own recordings**, `CLIPS_DIR/<folder>/<clip>.wav`, 16 kHz mono (not part of this repo) | none |
| `TRUTH_TSV` | what each recording really said: `clip<TAB>text<TAB>mark`, made with `label_clips.py` | `bench/truth.tsv` |
| `TESTSET_DIR` | the everyday test set, see `testset/README.md` | `bench/testset` |

| Script | What it measures | Needs | Backs this part of BENCHMARKS.md |
|---|---|---|---|
| `clip_length.py` | CPU, GPU and simple hybrid time on slices of `samples/jfk.wav` (2,500 to 15,000 ms); about 2 minutes | engine, GPU kit | speed against clip length |
| `wer_everyday.py` | CPU, GPU and the hybrid cut rules (live, balance, snap): word errors and time on the everyday test set; prints the table (the page quotes the copy in `results/wer_everyday.txt`; redirect to make your own, `> results/wer_everyday.txt`) | engine, GPU kit, `OIDO_TYPE`, test set audio | summary, cut rules |
| `four_ideas.py` | overlap stitching, balance on fresh curves, short-phrases-to-CPU, simulated streaming head; reads `results/wer_everyday.txt` first | same | four ideas |
| `stream_head.py` | the streaming head in real time on the everyday set | same | four ideas (idea 2) |
| `score_labelled.py` | CPU, GPU, hybrid, streaming head scored against your own text | recordings + `truth.tsv` | real recorded phrases |
| `speed_lab.py` | threads, priority and audio speed-up, with word errors, on your labelled clips | recordings + `truth.tsv` | other speed tests |
| `silence_rule.py` | how often a 500 / 400 ms phrase-end quiet splits your phrases, and the word errors | recordings + `truth.tsv`, `sherpa_onnx` | shorter quiet rule |
| `end_to_end.py cpu\|gpu\|hybrid` | replay of your recordings through the live voice detector: time from the end of speech until the text is ready and typed (env `CH` = mic chunk in samples, default 1280) | recordings + `truth.tsv`, `sherpa_onnx`, GPU kit | end to end |
| `label_clips.py` | the tool to write down what your recordings said (listen mode, or `--read` to correct the engine's text); `--make-engine-text` prepares that file | `CLIPS_DIR`, `mpv` | (makes `truth.tsv`) |

`results/` holds the output of the three everyday-set runs that the page quotes (numbers only). The scripts that use recordings print only counts and timings, never the text.

Not included: the stop-word check timing (it uses a separate wake-word model that is not in these repos), and the recordings and their labels (a person's voice).
Single runs on one board: see the Limits section of BENCHMARKS.md before quoting any number.

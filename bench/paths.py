"""Where the benchmark scripts find things. Every path comes from an environment variable, with a default that works from a clone of this repo.
See README.md for what each script needs."""
import os
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
def _p(env, default): return os.path.expanduser(os.environ.get(env, default))
OIDO_DIR = _p('OIDO_DIR', REPO)                                   # this repo (built: make -C host tasr_cli_a72; models downloaded into models/)
CLI = _p('OIDO_CLI', OIDO_DIR + '/host/tasr_cli_a72')              # the CPU engine (needs the --serve mode)
MODEL = _p('OIDO_MODEL', OIDO_DIR + '/models/nemo8.tnm')
LM = _p('OIDO_LM', OIDO_DIR + '/models/nemo_lm.tlm')
KIT = _p('GPU_PORT_KIT', '~/gpu-port-kit')                         # https://github.com/tarelerulz/gpu-port-kit (GPU engine)
OIDO_TYPE = _p('OIDO_TYPE', '~/oido-streaming/oido-type.py')       # https://github.com/tarelerulz/oido-streaming (HybridTail + cut rules)
SILERO = _p('SILERO_VAD', OIDO_DIR + '/models/silero_vad.onnx')    # Silero voice-activity model (used with sherpa_onnx)
_c = _p('CLIPS_DIR', '')                                           # your own recordings: CLIPS_DIR/<folder>/<clip>.wav (16 kHz mono)
CLIPS_DIR = (_c.rstrip('/') + '/') if _c else ''
TRUTH_FILE = _p('TRUTH_TSV', HERE + '/truth.tsv')                       # clip<TAB>correct text<TAB>mark (made with label_clips.py)
TESTSET = _p('TESTSET_DIR', HERE + '/testset')                     # sentences.json + clips16/ (see testset/README.md)
RESULTS_WER = _p('RESULTS_WER', HERE + '/results/wer_everyday.txt')  # output of wer_everyday.py, read by four_ideas.py and stream_head.py
JFK = _p('JFK_WAV', REPO + '/samples/jfk.wav')

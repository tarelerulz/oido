#!/usr/bin/env python3
"""Transcribe WAV files with Oido + punctuation/casing restoration.

usage: python transcribe.py [--model MODEL] [--lm LM] [--no-punct] file.wav ...

WAV files must be 16 kHz mono PCM16; convert with:
  ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
"""
import argparse, os, sys, time, wave
import numpy as np
import pytasr

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "..", "models")
PUNCT_MODEL = os.path.expanduser(
    "~/models/sherpa-onnx-online-punct-en-2024-08-06"
)


def read_wav(path):
    with wave.open(path) as w:
        if w.getnchannels() != 1 or w.getframerate() != 16000 or w.getsampwidth() != 2:
            sys.exit(f"{path}: need 16 kHz mono PCM16")
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def load_punct():
    try:
        import sherpa_onnx
        if not os.path.exists(PUNCT_MODEL):
            return None
        config = sherpa_onnx.OnlinePunctuationConfig(
            model_config=sherpa_onnx.OnlinePunctuationModelConfig(
                cnn_bilstm=os.path.join(PUNCT_MODEL, "model.int8.onnx"),
                bpe_vocab=os.path.join(PUNCT_MODEL, "bpe.vocab"),
            )
        )
        return sherpa_onnx.OnlinePunctuation(config)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--model", default=os.path.join(MODELS, "nemo8.tnm"))
    ap.add_argument("--lm", default=os.path.join(MODELS, "nemo_lm.tlm"))
    ap.add_argument("--no-punct", action="store_true")
    a = ap.parse_args()

    t0 = time.perf_counter()
    lm = pytasr.LM(a.lm) if os.path.exists(a.lm) else None
    model = pytasr.Nemo(a.model, lm=lm, beam=4 if lm else 0)
    punct = None if a.no_punct else load_punct()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"loaded in {load_ms:.0f}ms" + (" (no punct)" if not punct else ""), file=sys.stderr)

    for path in a.wavs:
        pcm = read_wav(path)
        dur = len(pcm) / 16000

        t0 = time.perf_counter()
        text = model.transcribe(pcm)
        asr_ms = (time.perf_counter() - t0) * 1000

        punct_ms = 0
        if punct and text:
            t0 = time.perf_counter()
            text = punct.add_punctuation_with_case(text)
            punct_ms = (time.perf_counter() - t0) * 1000

        print(text)
        print(f"  {dur:.1f}s audio | asr {asr_ms:.0f}ms | punct {punct_ms:.0f}ms | total {asr_ms+punct_ms:.0f}ms",
              file=sys.stderr)


if __name__ == "__main__":
    main()

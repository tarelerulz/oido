#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Hybrid CPU+GPU split of ONE long phrase (experiment).

A phrase of at least MIN_S seconds is cut at a real pause. The GPU encoder transcribes the head while a resident CPU engine
(tasr_cli --serve, low priority, cores 1-3) transcribes the tail at the same time; the two texts are joined. Any problem on the
CPU side returns None and the caller transcribes the whole phrase on the GPU as usual.

Typical use (gpu = any object with .transcribe(int16 pcm) -> str, e.g. the kit's Nemo8Encoder):

    hybrid = HybridTail(cli="host/tasr_cli", model="models/nemo8.tnm")
    cut = hybrid.split(pcm)                      # int16 mono 16 kHz, sample index or None
    if cut is not None:
        res = {}
        th = threading.Thread(target=lambda: res.setdefault("t", hybrid.tail(pcm[cut:])))
        th.start(); head = gpu.transcribe(pcm[:cut]).strip(); th.join(6.0)
        text = (head + " " + res["t"]).strip() if res.get("t") is not None else gpu.transcribe(pcm).strip()

See hybrid/README.md for measurements and caveats."""
import os
import subprocess
import tempfile
import threading
import wave
from pathlib import Path

import numpy as np

RATE = 16000


def _log(msg):
    if os.environ.get("OIDO_HYBRID_DEBUG"):
        print(f"hybrid: {msg}", flush=True)


class HybridTail:
    MIN_S = float(os.environ.get("OIDO_HYBRID_MIN", "4.5"))        # shorter phrases stay GPU-only
    RATIO = float(os.environ.get("OIDO_HYBRID_RATIO", "0.05"))     # cut only where 100 ms energy <= RATIO x the phrase's speech level
    # GPU time (ms) for a head of N seconds, measured on a Pi 400 at 800 MHz with the hand-written QPU matmul kernels
    # (steps at 64-frame boundaries). Used only to balance the cut; recalibrate for another GPU engine.
    GPU_PTS = [(0, 60), (1.5, 125), (2.4, 148), (3.1, 204), (3.7, 228), (4.5, 250), (5.8, 348), (7.0, 387), (9.4, 476), (10, 488),
               (15, 817), (19, 1127)]

    def __init__(self, cli, model, lm=None, cores="1,2,3", threads=3):
        """cli: tasr_cli binary, model: .tnm file, lm: optional language model (.tlm). The CPU engine loads in a background thread."""
        self.cli, self.model, self.lm, self.cores, self.threads = str(cli), str(model), lm, cores, threads
        self.p, self.lock, self.dead = None, threading.Lock(), False
        threading.Thread(target=self._start, daemon=True).start()

    def _start(self):
        try:
            # taskset needs a tasr_cli whose thread pool honours the inherited affinity mask (see components/tinyasr/tinyasr.c)
            cmd = ["nice", "-n", "19", "taskset", "-c", self.cores, self.cli, self.model]
            if self.lm:
                cmd += ["--lm", str(self.lm)]
            cmd += ["--serve"]
            env = {**os.environ, "TASR_THREADS": str(self.threads), "OMP_NUM_THREADS": str(self.threads)}
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, env=env)
            _log("resident CPU engine started")
        except Exception as e:
            self.dead = True
            _log(f"CPU engine failed to start, GPU only: {e}")

    def split(self, pcm):
        """Cut index (samples) or None. Cut at the quiet 100 ms window (35-75 % of the phrase) that balances the estimated GPU head
        and CPU tail times; None if there is no real pause or the estimated gain is under 10 %."""
        n = len(pcm)
        if self.dead or self.p is None or self.p.poll() is not None or n < int(self.MIN_S * RATE):
            return None
        xs, ys = [p[0] for p in self.GPU_PTS], [p[1] for p in self.GPU_PTS]
        gpu_ms = lambda sec: float(np.interp(sec, xs, ys))
        a = np.abs(pcm.astype(np.float32)); c = np.concatenate([[0.0], np.cumsum(a)]); w = RATE // 10
        st = np.arange(0, n - w, 160); e_all = (c[st + w] - c[st]) / w
        thr = max(30.0, self.RATIO * float(np.percentile(e_all, 70)))
        best = None
        for s0 in range(int(0.35 * n), int(0.75 * n), 160):
            if (c[s0 + w] - c[s0]) / w > thr:
                continue
            cut = s0 + w // 2
            t = max(1.08 * gpu_ms(cut / RATE), 1.1 * (10 + 45 * (n - cut) / RATE)) + 15    # GPU slows ~8 % and CPU ~10 % when both run
            if best is None or t < best[0]:
                best = (t, cut)
        if best is None or best[0] > 0.9 * gpu_ms(n / RATE):
            return None
        return int(best[1])

    def tail(self, pcm_tail):
        """Text of the tail via the CPU engine, or None on any problem (the caller then redoes the phrase on the GPU)."""
        with self.lock:
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", dir="/dev/shm" if os.path.isdir("/dev/shm") else None) as f:
                    with wave.open(f.name, "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
                        w.writeframes(pcm_tail.tobytes())
                    self.p.stdin.write(f.name + "\n"); self.p.stdin.flush()
                    out = {}
                    def rd():
                        out["l"] = self.p.stdout.readline()
                    th = threading.Thread(target=rd, daemon=True); th.start(); th.join(5.0)
                    if th.is_alive() or not out.get("l"):
                        raise RuntimeError("CPU engine did not answer")
                    return out["l"].rstrip("\n").split("\t")[-1].strip()
            except Exception as e:
                _log(f"CPU tail failed ({e}); disabled, GPU only")
                self.dead = True
                try:
                    self.p.kill()
                except Exception:
                    pass
                return None

    def close(self):
        try:
            if self.p is not None:
                self.p.stdin.close(); self.p.wait(timeout=5)
        except Exception:
            pass

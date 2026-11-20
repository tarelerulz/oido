#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Synthetic checks of HybridTail.split (no models, no audio files): python3 hybrid/test_hybrid_tail.py"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hybrid_tail import HybridTail, RATE


class Alive:
    def poll(self):
        return None


def make(dead=False):
    h = HybridTail.__new__(HybridTail)   # no CPU engine: split() only needs these fields
    h.dead, h.p = dead, Alive()
    return h


def speech(sec, rng):
    return (rng.standard_normal(int(sec * RATE)) * 3000).astype(np.int16)


def quiet(sec, rng):
    return (rng.standard_normal(int(sec * RATE)) * 5).astype(np.int16)


def main():
    rng = np.random.default_rng(1)
    h = make()
    # 8 s phrase: 4.6 s speech, 0.5 s pause, 2.9 s speech -> cut must fall inside the pause
    pcm = np.concatenate([speech(4.6, rng), quiet(0.5, rng), speech(2.9, rng)])
    cut = h.split(pcm)
    assert cut is not None, "pause not found"
    assert 4.6 * RATE <= cut <= 5.1 * RATE, f"cut {cut / RATE:.2f} s is outside the pause"
    # same length, no pause: no cut
    assert h.split(speech(8.0, rng)) is None, "cut without a pause"
    # shorter than MIN_S: no cut even with a pause
    short = np.concatenate([speech(1.5, rng), quiet(0.5, rng), speech(1.5, rng)])
    assert h.split(short) is None, "split a short phrase"
    # CPU engine dead: no cut
    assert make(dead=True).split(pcm) is None, "split with a dead CPU engine"
    print(f"ok: cut at {cut / RATE:.2f} s (pause 4.6-5.1 s)")


if __name__ == "__main__":
    main()

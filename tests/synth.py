"""Syntetické signály pro testy: pianový tón s nehармonickými parciálami, thump, šum."""
from __future__ import annotations
import numpy as np


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def piano_note(midi, sr, dur_s, peak=0.5, decay_db_s=-6.0, inharmonicity=0.0,
               n_partials=8, attack_ms=3.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(round(dur_s * sr))
    t = np.arange(n) / sr
    f0 = midi_to_hz(midi)
    x = np.zeros(n, dtype=np.float64)
    for h in range(1, n_partials + 1):
        fh = h * f0 * np.sqrt(1.0 + inharmonicity * h * h)
        if fh >= sr / 2:
            break
        # vyšší parciály doznívají rychleji (jako u struny)
        env = 10.0 ** ((decay_db_s * (1.0 + 0.3 * (h - 1))) * t / 20.0)
        x += (1.0 / h) * env * np.sin(2 * np.pi * fh * t + rng.uniform(0, 2 * np.pi))
    a = int(round(attack_ms / 1000.0 * sr))
    if a > 0:
        x[:a] *= np.linspace(0.0, 1.0, a)
    x *= peak / (np.abs(x).max() + 1e-12)
    return x.astype(np.float32)


def thump(sr, dur_ms=30.0, peak=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = int(round(dur_ms / 1000.0 * sr))
    t = np.arange(n) / sr
    x = rng.standard_normal(n) * np.exp(-t * (8.0 / (dur_ms / 1000.0)))
    x *= peak / (np.abs(x).max() + 1e-12)
    return x.astype(np.float32)


def noise_floor(n, sr, db=-80.0, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * 10.0 ** (db / 20.0)).astype(np.float32)


def place(canvas, x, at_s, sr):
    i = int(round(at_s * sr))
    j = min(len(canvas), i + len(x))
    canvas[i:j] += x[: j - i]

"""RMS obálka v dB a odvozené veličiny: lokální šumové dno, sklon dozvuku."""
from __future__ import annotations
import numpy as np

FLOOR_DB = -120.0


def hop_frames(sr: int, hop_ms: float) -> int:
    return max(1, int(round(sr * hop_ms / 1000.0)))


def rms_envelope_db(mono: np.ndarray, sr: int, hop_ms: float = 5.0) -> np.ndarray:
    hop = hop_frames(sr, hop_ms)
    n = len(mono) // hop
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    frames = mono[: n * hop].astype(np.float64).reshape(n, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(rms)
    return np.maximum(np.nan_to_num(db, nan=FLOOR_DB, neginf=FLOOR_DB), FLOOR_DB)


def smooth_db(env: np.ndarray, frames: int) -> np.ndarray:
    """Klouzavý medián lichého okna; kraje se doplní opakováním krajních hodnot."""
    if frames <= 1 or len(env) == 0:
        return env.copy()
    if frames % 2 == 0:
        frames += 1
    half = frames // 2
    padded = np.pad(env, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, frames)
    return np.median(windows, axis=1)


def local_floor_db(env: np.ndarray, i_onset: int, hop_s: float, window_s: float = 2.0) -> float:
    w = int(round(window_s / hop_s))
    seg = env[max(0, i_onset - w): i_onset]
    if len(seg) == 0:
        return float(env.min()) if len(env) else FLOOR_DB
    return float(np.percentile(seg, 5))


def decay_slope_db_s(env: np.ndarray, i_end: int, hop_s: float, window_s: float = 2.0) -> float:
    w = int(round(window_s / hop_s))
    seg = env[max(0, i_end - w): i_end]
    if len(seg) < 2:
        return 0.0
    t = np.arange(len(seg)) * hop_s
    slope, _ = np.polyfit(t, seg, 1)
    return float(slope)

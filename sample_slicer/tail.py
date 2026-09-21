"""Konec samplu: přirozený exponenciální dozvuk do nuly, fade-in proti kliku."""
from __future__ import annotations
import numpy as np
from .detect import Segment


def apply_fade_in(data: np.ndarray, sr: int, ms: float = 2.0) -> np.ndarray:
    n = min(len(data), int(round(ms / 1000.0 * sr)))
    if n > 0:
        data[:n] *= np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)[:, None]
    return data


def natural_tail(data: np.ndarray, sr: int, slope_db_s: float, tail_s: float = 2.0,
                 floor_db: float = -96.0, final_ms: float = 10.0) -> np.ndarray:
    """Na posledních `tail_s` aplikuje exponenciální útlum navazující na naměřený sklon dozvuku:
    strmost = max(|slope|, |floor_db|/tail_s) dB/s; posledních `final_ms` lineárně do nuly."""
    out = data.copy()
    n_tail = min(len(out), int(round(tail_s * sr)))
    if n_tail == 0:
        return out
    rate = max(abs(slope_db_s), abs(floor_db) / tail_s)     # dB/s, vždy kladné
    t = np.arange(n_tail) / sr
    gain = (10.0 ** (-rate * t / 20.0)).astype(np.float32)
    n_fin = min(n_tail, int(round(final_ms / 1000.0 * sr)))
    if n_fin > 0:
        gain[-n_fin:] *= np.linspace(1.0, 0.0, n_fin, dtype=np.float32)
    out[-n_tail:] *= gain[:, None]
    return out


def render_segment(audio: np.ndarray, sr: int, seg: Segment, tail_s: float = 2.0,
                   fade_in_ms: float = 2.0) -> np.ndarray:
    """Vyřízne audio[seg.start : seg.end + tail_s] (2D), fade-in, přirozený dozvuk; za EOF doplní nuly."""
    n_tail = int(round(tail_s * sr))
    stop = seg.end + n_tail
    if seg.limit > 0:
        stop = min(stop, seg.limit)              # dozvuk nikdy nesahá za nasazení dalšího tónu
    piece = audio[seg.start: min(stop, len(audio))].astype(np.float32, copy=True)
    if stop > len(audio):
        piece = np.concatenate([piece, np.zeros((stop - len(audio), audio.shape[1]), dtype=np.float32)])
    apply_fade_in(piece, sr, fade_in_ms)
    return natural_tail(piece, sr, seg.slope_db_s, tail_s)

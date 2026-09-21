"""Čtení a zápis WAV (PCM int 16/24/32) přes standardní `wave`, data jako float32."""
from __future__ import annotations
import struct, wave
from dataclasses import dataclass
from pathlib import Path
import numpy as np


class UnsupportedWav(Exception):
    """WAV není PCM int 16/24/32 bit."""


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    channels: int
    bits: int
    frames: int


def _check_pcm_int(path: Path) -> None:
    # `wave` float WAV buď odmítne, nebo mylně přijme; fmt tag ověříme ručně.
    with open(path, "rb") as f:
        head = f.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise UnsupportedWav(f"{path}: není RIFF/WAVE")
        while True:
            chunk = f.read(8)
            if len(chunk) < 8:
                raise UnsupportedWav(f"{path}: chybí fmt chunk")
            cid, size = chunk[:4], struct.unpack("<I", chunk[4:])[0]
            if cid == b"fmt ":
                body = f.read(size)
                fmt_tag = struct.unpack("<H", body[:2])[0]
                if fmt_tag == 0xFFFE and len(body) >= 26:
                    # WAVE_FORMAT_EXTENSIBLE: skutečný formát je v SubFormat GUID (offset 24 v těle chunku)
                    fmt_tag = struct.unpack("<H", body[24:26])[0]
                if fmt_tag != 1:
                    raise UnsupportedWav(f"{path}: formát {fmt_tag} není PCM int")
                return
            f.seek(size + (size & 1), 1)


def read_wav(path) -> tuple[np.ndarray, WavInfo]:
    path = Path(path)
    _check_pcm_int(path)
    with wave.open(str(path), "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    bits = sw * 8
    if bits == 16:
        ints = np.frombuffer(raw, dtype="<i2").astype(np.int32)
    elif bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        ints = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
                | (b[:, 2].astype(np.int8).astype(np.int32) << 16))
    elif bits == 32:
        ints = np.frombuffer(raw, dtype="<i4")
    else:
        raise UnsupportedWav(f"{path}: {bits} bit není podporováno")
    data = (ints.astype(np.float32) / float(2 ** (bits - 1))).reshape(-1, ch)
    return data, WavInfo(sr, ch, bits, data.shape[0])


def write_wav(path, data: np.ndarray, sample_rate: int, bits: int) -> None:
    if data.ndim != 2:
        raise ValueError("data musí mít tvar (frames, channels)")
    if bits not in (16, 24, 32):
        raise UnsupportedWav(f"{bits} bit není podporováno")
    full = float(2 ** (bits - 1))
    ints = np.clip(np.round(data.astype(np.float64) * full), -full, full - 1).astype(np.int32)
    if bits == 16:
        raw = ints.astype("<i2").tobytes()
    elif bits == 24:
        flat = ints.reshape(-1)
        b = np.empty((flat.size, 3), dtype=np.uint8)
        b[:, 0] = flat & 0xFF
        b[:, 1] = (flat >> 8) & 0xFF
        b[:, 2] = (flat >> 16) & 0xFF
        raw = b.tobytes()
    else:
        raw = ints.astype("<i4").tobytes()
    with wave.open(str(path), "wb") as w:
        w.setnchannels(data.shape[1])
        w.setsampwidth(bits // 8)
        w.setframerate(sample_rate)
        w.writeframes(raw)


def remove_dc(data: np.ndarray) -> np.ndarray:
    return data - data.mean(axis=0, keepdims=True)


def to_mono(data: np.ndarray) -> np.ndarray:
    return data.mean(axis=1)

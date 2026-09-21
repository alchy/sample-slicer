"""Generický střih: soubor → seznam ořezaných úderů (bez znalosti not)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .io import read_wav, write_wav, remove_dc, to_mono, UnsupportedWav
from .detect import detect_segments, DetectParams, Segment
from .tail import render_segment


@dataclass
class Slice:
    index: int
    segment: Segment
    audio: np.ndarray
    sr: int
    bits: int
    channels: int
    source: Path


def slice_file(path, params: DetectParams = DetectParams(), tail_s: float = 2.0,
               fade_in_ms: float = 2.0) -> tuple[list[Slice], list[Segment]]:
    path = Path(path)
    data, info = read_wav(path)
    data = remove_dc(data)
    accepted, rejected = detect_segments(to_mono(data), info.sample_rate, params)
    slices = [Slice(i, seg, render_segment(data, info.sample_rate, seg, tail_s, fade_in_ms),
                    info.sample_rate, info.bits, info.channels, path)
              for i, seg in enumerate(accepted)]
    return slices, rejected


def slice_dir(in_dir, out_dir, params: DetectParams = DetectParams(), tail_s: float = 2.0,
              fade_in_ms: float = 2.0, log=print) -> int:
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for wav in sorted(in_dir.glob("*.[wW][aA][vV]")):
        try:
            slices, rejected = slice_file(wav, params, tail_s, fade_in_ms)
        except UnsupportedWav as e:
            log(f"PŘESKOČENO {wav.name}: {e}")
            continue
        for s in slices:
            start_ms = int(s.segment.start / s.sr * 1000)
            dur_ms = int(len(s.audio) / s.sr * 1000)
            name = f"{wav.stem}_slice_{s.index + 1:03d}_start_{start_ms}ms_dur_{dur_ms}ms.wav"
            write_wav(out_dir / name, s.audio, s.sr, s.bits)
            count += 1
        log(f"{wav.name}: {len(slices)} úderů, {len(rejected)} kliků zahozeno")
    return count

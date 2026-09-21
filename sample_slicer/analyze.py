"""Dry-run analýza: segmenty + výška + přiřazení; s pravdou i přesnost po oktávách."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from .detect import DetectParams, Segment
from .io import UnsupportedWav, to_mono
from .notes import expand_truth, midi_to_name
from .pitch import Pitch, estimate_pitch
from .slicing import slice_file, Slice
from .tuning import Assignment, assign_notes, TuningParams


@dataclass
class HitRow:
    source: str
    index: int
    t_s: float
    dur_s: float
    peak_db: float
    end_reason: str
    pitch: Pitch | None
    assignment: Assignment
    truth: int | None


def analyze_file(path, params: DetectParams, tail_s: float, truth_entry: dict | None,
                 tuning: TuningParams = TuningParams()) -> tuple[list[HitRow], list[Segment], list[Slice]]:
    """Vrátí (řádky, odmítnuté kliky, slices) — slices se vrací, aby build nemusel číst zdroj dvakrát."""
    slices, rejected = slice_file(path, params, tail_s)
    pitches = [estimate_pitch(to_mono(s.audio)[s.segment.onset - s.segment.start:], s.sr) for s in slices]
    assignments, _ = assign_notes(pitches, tuning)
    truths = expand_truth(truth_entry, len(slices)) if truth_entry else [None] * len(slices)
    rows = [HitRow(Path(path).name, s.index, s.segment.onset / s.sr, len(s.audio) / s.sr, s.segment.peak_db,
                   s.segment.end_reason, pt, a, tr)
            for s, pt, a, tr in zip(slices, pitches, assignments, truths)]
    return rows, rejected, slices


def analyze_dir(src_dir, params: DetectParams, tail_s: float, truth: dict | None, log=print) -> list[HitRow]:
    rows: list[HitRow] = []
    for wav in sorted(Path(src_dir).glob("*.[wW][aA][vV]")):
        entry = (truth or {}).get(wav.name)
        try:
            r, rejected, _ = analyze_file(wav, params, tail_s, entry)
        except UnsupportedWav as e:
            log(f"PŘESKOČENO {wav.name}: {e}")
            continue
        log(f"{wav.name}: {len(r)} úderů, {len(rejected)} kliků")
        rows.extend(r)
    return rows


def _octave_key(midi: int) -> str:
    lo = 21 + 12 * ((midi - 21) // 12)
    return f"{midi_to_name(lo)}-{midi_to_name(min(lo + 11, 108))}"


def accuracy_by_octave(rows):
    acc: dict[str, dict[str, int]] = {}
    for r in rows:
        if r.truth is None:
            continue
        d = acc.setdefault(_octave_key(r.truth), {"ok": 0, "octave": 0, "other": 0})
        got = r.assignment.midi
        if got == r.truth:
            d["ok"] += 1
        elif got is not None and (got - r.truth) % 12 == 0:
            d["octave"] += 1
        else:
            d["other"] += 1
    return acc


def format_rows(rows) -> str:
    lines = [f"{'zdroj':<22}{'#':>3}{'čas':>9}{'délka':>7}{'peak':>7}{'konec':>11}{'nota':>6}{'midi':>8}{'c/ET':>6}{'c/křivka':>9}{'conf':>6}  verdikt"]
    for r in rows:
        pt = r.pitch
        note = midi_to_name(r.assignment.midi) if r.assignment.midi is not None else "-"
        lines.append(f"{r.source:<22}{r.index:>3}{r.t_s:>9.2f}{r.dur_s:>7.1f}{r.peak_db:>7.1f}{r.end_reason:>11}"
                     f"{note:>6}{(pt.midi if pt else float('nan')):>8.2f}{r.assignment.cents_et:>6.0f}"
                     f"{r.assignment.cents_curve:>9.0f}{(pt.confidence if pt else 0):>6.2f}  {r.assignment.reason}"
                     + (f"  truth {midi_to_name(r.truth)}" if r.truth is not None else ""))
    return "\n".join(lines)

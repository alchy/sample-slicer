"""Názvy not ↔ MIDI a rozbalení „pravdy" (známé pořadí nahrávání) na posloupnost MIDI."""
from __future__ import annotations
import re

_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_SEMI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d+)$")
PATTERNS = {"chromatic": [1], "major": [2, 2, 1, 2, 2, 2, 1]}


def note_to_midi(name: str) -> int:
    m = _RE.match(name.strip())
    if not m:
        raise ValueError(f"neplatný název noty: {name!r}")
    letter, acc, octave = m.group(1).upper(), m.group(2), int(m.group(3))
    return 12 * (octave + 1) + _SEMI[letter] + (1 if acc == "#" else -1 if acc == "b" else 0)


def midi_to_name(midi: int) -> str:
    return f"{_NAMES[midi % 12]}{midi // 12 - 1}"


def expand_truth(entry: dict, count: int) -> list[int]:
    pattern = entry.get("pattern", "chromatic")
    if pattern == "list":
        return [note_to_midi(n) for n in entry["notes"]]
    if pattern not in PATTERNS:
        raise ValueError(f"neznámý pattern {pattern!r}; podporováno: chromatic, major, list")
    steps = PATTERNS[pattern]
    out = [note_to_midi(entry["start"])]
    while len(out) < count:
        out.append(out[-1] + steps[(len(out) - 1) % len(steps)])
    return out[:count]

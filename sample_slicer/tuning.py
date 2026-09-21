"""Ladicí křivka piana (odchylka v centech vs. MIDI) a postupné přiřazení not.

Piano není temperované přesně (Petrof: bas -30 až +9 c, střed ±20 c, A7–C8 +46 až +79 c).
Tón +79 c je k nerozeznání od sousední noty -21 c, proto se noty přiřazují postupně:
nejdřív jisté kotvy (blízko celé noty a konzistentní s křivkou z ostatních kotev), pak
zbytek od nejjistějšího vůči křivce, která se po každém přiřazení přepočítá. Křivka
mimo kotvy extrapoluje lineárně (stretch v krajních oktávách roste strmě)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from .pitch import Pitch


@dataclass
class TuningParams:
    anchor_cents: float = 35.0
    accept_cents: float = 50.0
    min_confidence: float = 0.5
    curve_halfwidth: int = 2
    max_slope_cents: float = 25.0      # strop sklonu extrapolace (c / půltón)
    midi_lo: int = 21
    midi_hi: int = 108


@dataclass
class Assignment:
    midi: int | None
    cents_et: float
    cents_curve: float
    anchor: bool
    reason: str


def fit_tuning_curve(anchors: list[tuple[int, float]], halfwidth: int = 2,
                     max_slope: float = 25.0) -> Callable[[float], float]:
    """Kotvy (midi, centy) → funkce midi → centy: klouzavý medián ±halfwidth půltónů,
    lineární interpolace mezi kotvami, lineární extrapolace z krajních dvou bodů
    (sklon omezen na ±max_slope); s jednou kotvou konstanta, bez kotev 0."""
    if not anchors:
        return lambda m: 0.0
    by_midi: dict[int, list[float]] = {}
    for m, c in anchors:
        by_midi.setdefault(m, []).append(c)
    xs = np.array(sorted(by_midi), dtype=float)
    raw = np.array([np.median(by_midi[int(m)]) for m in xs])
    ys = np.array([np.median(raw[(xs >= m - halfwidth) & (xs <= m + halfwidth)]) for m in xs])

    def slope(i0, i1):
        if xs[i1] == xs[i0]:
            return 0.0
        return float(np.clip((ys[i1] - ys[i0]) / (xs[i1] - xs[i0]), -max_slope, max_slope))

    def curve(m: float) -> float:
        if len(xs) == 1:
            return float(ys[0])
        if m < xs[0]:
            return float(ys[0] + slope(0, 1) * (m - xs[0]))
        if m > xs[-1]:
            return float(ys[-1] + slope(-2, -1) * (m - xs[-1]))
        return float(np.interp(m, xs, ys))
    return curve


def _split_note(midi_float: float, curve) -> tuple[int, float, float]:
    """Vrátí (nota, centy vs. temperované, centy vs. křivka) pro nejbližší notu po odečtení křivky."""
    corrected = midi_float - curve(midi_float) / 100.0
    m = int(np.floor(corrected + 0.5))
    return m, 100.0 * (midi_float - m), 100.0 * (corrected - m)


def assign_notes(pitches: list[Pitch | None], p: TuningParams = TuningParams()):
    n = len(pitches)
    out: list[Assignment | None] = [None] * n
    usable = []
    for i, pt in enumerate(pitches):
        if pt is None or pt.confidence < p.min_confidence:
            out[i] = Assignment(None, 0.0, 0.0, False, "low_confidence")
        else:
            usable.append(i)

    # 1. průchod: kandidáti na kotvy = blízko celé noty a v rozsahu
    cand = {}
    for i in usable:
        pt = pitches[i]
        m = int(np.floor(pt.midi + 0.5))
        if p.midi_lo <= m <= p.midi_hi and abs(pt.cents_et) <= p.anchor_cents:
            cand[i] = (m, pt.cents_et)
    # kotva musí být konzistentní s křivkou z OSTATNÍCH kotev (leave-one-out);
    # jinak je to nejspíš sousední nota s velkou odchylkou
    anchors: dict[int, tuple[int, float]] = {}
    for i, (m, c) in cand.items():
        others = [v for j, v in cand.items() if j != i]
        if not others or abs(c - fit_tuning_curve(others, p.curve_halfwidth, p.max_slope_cents)(m)) <= p.anchor_cents:
            anchors[i] = (m, c)
    for i, (m, c) in anchors.items():
        out[i] = Assignment(m, c, 0.0, True, "anchor")

    # 2. průchod: postupně, vždy nejjistější zbývající úder vůči aktuální křivce
    remaining = [i for i in usable if out[i] is None]
    while remaining:
        curve = fit_tuning_curve(list(anchors.values()), p.curve_halfwidth, p.max_slope_cents)
        best = min(remaining, key=lambda i: abs(_split_note(pitches[i].midi, curve)[2]))
        m, cents_et, cents_curve = _split_note(pitches[best].midi, curve)
        remaining.remove(best)
        if not (p.midi_lo <= m <= p.midi_hi):
            out[best] = Assignment(None, cents_et, cents_curve, False, "out_of_range")
        elif abs(cents_curve) >= p.accept_cents:
            out[best] = Assignment(None, cents_et, cents_curve, False, "out_of_tolerance")
            # tenhle se nepřidá do kotev, ale zbytek se ještě zkusí
        else:
            out[best] = Assignment(m, cents_et, cents_curve, False, "curve")
            anchors[best] = (m, cents_et)
    curve = fit_tuning_curve(list(anchors.values()), p.curve_halfwidth, p.max_slope_cents)
    for i, a in enumerate(out):
        if a is not None and a.anchor:
            out[i] = Assignment(a.midi, a.cents_et, a.cents_et - curve(a.midi), True, "anchor")
    return out, curve

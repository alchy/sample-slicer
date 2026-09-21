"""Odhad výšky tónu: autokorelace s normalizací pásma („zrychlení" přes k), hlasování,
spektrální důkaz parciál, doladění na fundamentálu z plného sample rate.
Ověřeno na reálných nahrávkách Petrof (A0–G#2 24/24, C4–C8 27/29 + ladicí křivka)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def midi_from_hz(f: float) -> float:
    return 69.0 + 12.0 * np.log2(f / 440.0)


def hz_from_midi(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69.0) / 12.0)


@dataclass
class Pitch:
    f0_hz: float
    midi: float
    confidence: float
    n_votes: int
    evidence: int

    @property
    def cents_et(self) -> float:
        return 100.0 * (self.midi - round(self.midi))


def _highpass(x: np.ndarray, sr: float, fc: float = 20.0) -> np.ndarray:
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1.0 / sr)
    X[f < fc] = 0.0
    return np.fft.irfft(X, len(x))


def _acf_peak(x: np.ndarray, sr: float, fmin: float, fmax: float):
    """Normalizovaná ACF; vrátí (f, výška) nejvyššího VNITŘNÍHO lokálního maxima v pásmu, nebo None."""
    x = x - x.mean()
    n = len(x)
    if n < 8 or not np.any(x):
        return None
    X = np.fft.rfft(x, 2 * n)
    r = np.fft.irfft(X * np.conj(X))[:n]
    if r[0] <= 0:
        return None
    r = r / r[0]
    lmin, lmax = int(sr / fmax), min(int(sr / fmin), n - 2)
    if lmax - lmin < 3:
        return None
    seg = r[lmin:lmax]
    loc = np.where((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]))[0] + 1
    if len(loc) == 0:
        return None
    i = int(loc[np.argmax(seg[loc])]) + lmin
    a, b, c = r[i - 1], r[i], r[i + 1]
    denom = a - 2 * b + c
    d = 0.5 * (a - c) / denom if denom != 0 else 0.0
    return sr / (i + d), float(b)


def _spectrum(x: np.ndarray, sr: int, zero_pad: int = 8):
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(max(16, n * zero_pad))))
    mag = np.abs(np.fft.rfft(x * np.hanning(n), nfft))
    return mag, sr / nfft


def _prominence_db(mag, df, f0):
    """Ostrost spektrálního vrcholu u f0: max v ±60 c minus medián log-spektra v ±1 oktávě."""
    lo, hi = int(f0 * 2 ** (-60 / 1200) / df), int(f0 * 2 ** (60 / 1200) / df) + 1
    olo, ohi = int(f0 / 2 / df), min(int(f0 * 2 / df), len(mag) - 1)
    if hi >= len(mag) or olo >= ohi or lo < 1:
        return -99.0
    L = 20 * np.log10(mag + 1e-12)
    return float(L[lo:hi].max() - np.median(L[olo:ohi]))


def _peak_near(mag, df, c, tol_cents):
    lo, hi = int(c * 2 ** (-tol_cents / 1200) / df), int(c * 2 ** (tol_cents / 1200) / df) + 1
    if hi >= len(mag) - 1 or lo < 1:
        return None
    i = lo + int(np.argmax(mag[lo:hi]))
    a, b, cc = np.log(mag[i - 1] + 1e-12), np.log(mag[i] + 1e-12), np.log(mag[i + 1] + 1e-12)
    denom = a - 2 * b + cc
    d = 0.5 * (a - cc) / denom if denom != 0 else 0.0
    return (i + d) * df


def estimate_pitch(x, sr, offset_s=0.04, win_nom_s=0.25, ks=(0.25, 0.5, 1, 2, 4, 8, 16),
                   fmin_nom=100.0, fmax_nom=1600.0, min_peak=0.6, cluster_cents=50.0,
                   prominence_db=12.0, min_evidence=2, refine_cents=150.0):
    """x = 1D mono úsek začínající nasazením. Vrací Pitch nebo None (žádný použitelný vrchol)."""
    x = np.asarray(x, dtype=np.float64)
    o = int(round(offset_s * sr))
    votes = []
    for k in ks:
        srn = sr * k
        seg = x[o: o + int(win_nom_s * srn)]
        if len(seg) < int(0.1 * srn):
            continue
        p = _acf_peak(_highpass(seg, srn), srn, fmin_nom, fmax_nom)
        if p and p[1] >= min_peak:
            votes.append((p[0] / k, p[1], k))
    if not votes:
        return None
    spec_seg = _highpass(x[o: o + int(0.5 * sr)], sr)
    if len(spec_seg) < 64:
        return None
    mag, df = _spectrum(spec_seg, sr)
    clusters = []   # (f, score, n, evidence, proms)
    for f, pk, k in votes:
        if any(abs(1200 * np.log2(c[0] / f)) < cluster_cents for c in clusters):
            continue
        mem = [v for v in votes if abs(1200 * np.log2(v[0] / f)) < cluster_cents]
        fm = float(np.median([v[0] for v in mem]))
        proms = [_prominence_db(mag, df, fm * h) for h in (1, 2, 3, 4)]
        ev = sum(p >= prominence_db for p in proms)
        score = sum(v[1] ** 2 for v in mem) * (1.0 if ev >= min_evidence else 0.1)
        clusters.append((fm, score, len(mem), ev, proms))
    fm, score, n, ev, proms = max(clusters, key=lambda c: c[1])
    total = sum(c[1] for c in clusters)
    f0 = fm
    for h, p in zip((1, 2, 3, 4), proms):
        if p >= prominence_db:
            fr = _peak_near(mag, df, fm * h, refine_cents)
            if fr:
                f0 = fr / h
            break
    return Pitch(f0_hz=float(f0), midi=float(midi_from_hz(f0)), confidence=float(score / total),
                 n_votes=n, evidence=ev)

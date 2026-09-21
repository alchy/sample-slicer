"""Detekce úderů v nahrávce: nasazení, dělení slitých tónů, konec, artefakt uvolnění, kliky."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .envelope import rms_envelope_db, smooth_db, local_floor_db, decay_slope_db_s, hop_frames


@dataclass
class DetectParams:
    hop_ms: float = 5.0
    onset_rise_db: float = 20.0
    onset_rise_ms: float = 30.0
    preroll_ms: float = 5.0
    split_rise_db: float = 12.0
    split_peak_within_db: float = 20.0
    split_min_len_s: float = 0.5
    click_below_peak_db: float = 25.0
    end_level_db: float = -60.0
    artifact_after_s: float = 1.0
    artifact_window_s: float = 2.0
    artifact_rise_db: float = 8.0
    max_len_s: float = 30.0
    smooth_ms: float = 100.0
    min_len_s: float = 0.3


@dataclass
class Segment:
    start: int
    onset: int
    end: int
    peak_db: float
    floor_db: float
    slope_db_s: float
    end_reason: str


def _onset_frame(env: np.ndarray, i_rise: int, i: int) -> int:
    """Nasazení = první rámec v (i_rise, i], který je zřetelně (6 dB) nad úrovní před skokem.
    Zpětné hledání minima v šumu by skákalo; pre-roll 5 ms doplní zbytek."""
    pre = env[i_rise]
    for j in range(i_rise + 1, i + 1):
        if env[j] > pre + 6.0:
            return j
    return i


def find_onsets(env: np.ndarray, hop_s: float, p: DetectParams) -> list[int]:
    rise_frames = max(1, int(round(p.onset_rise_ms / 1000.0 / hop_s)))
    n = len(env)
    onsets: list[int] = []
    i = rise_frames
    active_peak = None      # vrchol aktuálního úderu (pro pravidlo slitých tónů)
    while i < n:
        rise = env[i] - env[i - rise_frames]
        if active_peak is None:
            floor = local_floor_db(env, i - rise_frames, hop_s)
            if rise > p.onset_rise_db and env[i] > floor + p.onset_rise_db:
                onsets.append(_onset_frame(env, i - rise_frames, i))
                active_peak = env[i]
                i += rise_frames
                continue
        else:
            active_peak = max(active_peak, env[i])
            if rise > p.split_rise_db:
                # kandidát na nový úder uvnitř aktivního úseku
                look = int(round(p.split_min_len_s / hop_s))
                new_peak = env[i: i + look].max() if i + 1 < n else env[i]
                sustained = (i + look <= n) and (env[i: i + look].min() > env[i - rise_frames] - 3.0)
                if new_peak >= active_peak - p.split_peak_within_db and sustained:
                    onsets.append(_onset_frame(env, i - rise_frames, i))
                    active_peak = new_peak
                    i += rise_frames
                    continue
            # konec aktivního úseku = obálka spadla k dnu; pak se zase hledá nasazení
            floor = local_floor_db(env, onsets[-1], hop_s)
            if env[i] < floor + 6.0:
                active_peak = None
        i += 1
    return onsets


def _segment_end(env: np.ndarray, env_s: np.ndarray, i_onset: int, i_limit: int, hop_s: float, p: DetectParams):
    """Vrátí (i_end, reason, slope). Konec podle úrovně čte vyhlazenou obálku env_s; artefakt
    uvolnění (krátký thump) hledá v surové obálce env proti regresní čáře z env_s.
    i_limit = začátek dalšího úderu nebo len."""
    i_max = min(i_limit, i_onset + int(round(p.max_len_s / hop_s)))
    i_peak = i_onset + int(np.argmax(env_s[i_onset: i_max])) if i_max > i_onset else i_onset
    after = i_onset + int(round(p.artifact_after_s / hop_s))
    win = int(round(p.artifact_window_s / hop_s))
    i = max(i_peak + 1, after)
    while i < i_max:
        if env_s[i] < p.end_level_db:
            return i, "level", decay_slope_db_s(env_s, i, hop_s, p.artifact_window_s)
        if i - after >= win:
            seg = env_s[i - win: i]
            t = np.arange(win) * hop_s
            slope, icpt = np.polyfit(t, seg, 1)
            predicted = icpt + slope * win * hop_s
            if env[i] - predicted > p.artifact_rise_db:
                back = int(round(0.1 / hop_s))
                j = i - back + int(np.argmin(env_s[i - back: i])) if back > 0 else i
                return j, "artifact", float(slope)
        i += 1
    if i_limit < len(env_s) and i_max == i_limit:
        # konec u dalšího úderu: sklon měř před náběhem (vyhlazení by ho jinak zahrnulo)
        guard = int(round(p.smooth_ms / 1000.0 / hop_s))
        return i_max, "next_onset", decay_slope_db_s(env_s, max(i_onset + 1, i_max - guard), hop_s, p.artifact_window_s)
    reason = "max_len" if i_max < i_limit else "eof"
    return i_max, reason, decay_slope_db_s(env_s, i_max, hop_s, p.artifact_window_s)


def detect_segments(mono: np.ndarray, sr: int, p: DetectParams = DetectParams()):
    hop = hop_frames(sr, p.hop_ms)
    hop_s = hop / sr
    env = rms_envelope_db(mono, sr, p.hop_ms)
    env_s = smooth_db(env, max(1, int(round(p.smooth_ms / 1000.0 / hop_s))))
    onsets = find_onsets(env, hop_s, p)
    preroll = int(round(p.preroll_ms / 1000.0 * sr))
    segs: list[Segment] = []
    for k, i_on in enumerate(onsets):
        i_limit = onsets[k + 1] if k + 1 < len(onsets) else len(env_s)
        i_end, reason, slope = _segment_end(env, env_s, i_on, i_limit, hop_s, p)
        onset = i_on * hop
        start = max(0, onset - preroll)
        end = min(len(mono), i_end * hop)
        if reason == "next_onset":
            end = min(end, max(onset, i_limit * hop - preroll))   # nesmí sahat do pre-rollu dalšího úderu
        if (end - onset) / sr < p.min_len_s:
            continue
        peak = float(env[i_on: max(i_on + 1, i_end)].max())
        segs.append(Segment(start, onset, end, peak, local_floor_db(env, i_on, hop_s), slope, reason))
    if not segs:
        return [], []
    loudest = max(s.peak_db for s in segs)
    accepted = [s for s in segs if s.peak_db >= loudest - p.click_below_peak_db]
    rejected = [s for s in segs if s.peak_db < loudest - p.click_below_peak_db]
    return accepted, rejected

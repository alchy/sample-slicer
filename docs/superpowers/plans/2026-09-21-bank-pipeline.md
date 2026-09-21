# Bank pipeline (nahrávka → banka samplů) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Z adresáře surových nahrávek piana (96 kHz / 24 bit) postavit jedním příkazem dynamic-velocity banku pro ithaca-legacy: ořezané údery s přirozeným dozvukem, nota určená jen z audia, výstup v originálu i v 48 kHz / 16 bit, s indexem a reportem.

**Architecture:** Stávající `slicer.py` se refaktoruje na balíček `sample_slicer` s malými moduly (io, envelope, detect, tail, pitch, tuning, bank, cli). Každý modul pracuje nad numpy poli a je testovatelný bez souborů. CLI `sample-slicer` má podpříkazy `slice` (původní generické chování), `analyze` (dry-run + měření přesnosti proti známé pravdě) a `build` (celý workflow). GUI (`slicergui`) volá stejné API.

**Tech Stack:** Python ≥ 3.11, numpy, tqdm, pytest; externí ffmpeg (soxr) pro převod na 48 kHz / 16 bit.

**Spec:** `docs/superpowers/specs/2026-09-21-bank-pipeline-design.md` (čti spolu s plánem; plán z něj argumentuje).

## Global Constraints

- Python ≥ 3.11; závislosti jen `numpy`, `tqdm` (+ `pytest` pro vývoj); GUI (`PySide6`, `platformdirs`) je volitelný extra.
- Vstupní WAV: PCM int 16/24/32 bit, mono nebo stereo, libovolný sample rate; float WAV → chyba pro daný soubor, pokračuje se.
- Analýza vždy na mono mixu `0.5*(L+R)` po odečtení DC offsetu per kanál; výstup zachovává původní počet kanálů, bit depth i sample rate.
- Výchozí prahy (parametry, každý přepsatelný): nasazení +20 dB nad lokální dno během 30 ms; pre-roll 5 ms; fade-in 2 ms; dělení slitých tónů: skok 12 dB / 30 ms, vrchol do 20 dB od předchozího, délka ≥ 0,5 s; kliky: vrchol > 25 dB pod nejhlasitějším úderem souboru; `end_level` -60 dBFS; artefakt uvolnění: > 8 dB nad regresní čarou (okno 2 s) po 1 s dozvuku; max délka 30 s; `tail_s` 2 s do -96 dB, posledních 10 ms lineárně do nuly.
- Pitch: k ∈ {¼, ½, 1, 2, 4, 8, 16}, okno 250 ms nominálně, pásmo 100–1600 Hz nominálně, vrchol ACF ≥ 0,6 a vnitřní lokální maximum, shluky ±50 c, spektrální důkaz: ≥ 2 z parciál 1–4 s prominencí ≥ 12 dB, jinak skóre ×0,1; doladění ±150 c kolem nejnižší prominentní parciály, plné sample rate.
- Přiřazení: kotvy ±35 c od celé noty; křivka = klouzavý medián ±6 půltónů, lineární interpolace, konstantní extrapolace; přijetí ±50 c od křivky; rozsah MIDI 21–108.
- Názvy souborů v bance: prvních 16 hex znaků MD5 obsahu; layout `m###/<hash>.wav`; žádná normalizace hlasitosti.
- ffmpeg převod: `-af aresample=48000:resampler=soxr:precision=28:dither_method=triangular -c:a pcm_s16le`.
- Nic se nemaže mimo soubory, které index sám vytvořil. Nikdy tiché přiřazení noty: nízká confidence / mimo toleranci → `_rejected/`.
- Komentáře v kódu česky (repo sample-slicer používá diakritiku), identifikátory anglicky.
- Testy: `.venv/bin/pytest -q` musí projít po každém tasku. Commit po každém tasku.

---

## Mapa souborů

| Soubor | Odpovědnost |
|---|---|
| `pyproject.toml` | balíček `sample_slicer`, konzolový příkaz `sample-slicer`, extras `gui`, `dev` |
| `sample_slicer/__init__.py` | `__version__` |
| `sample_slicer/io.py` | `read_wav`, `write_wav`, `remove_dc`, `to_mono` |
| `sample_slicer/envelope.py` | `rms_envelope_db`, `smooth_db`, `local_floor_db`, `decay_slope_db_s` |
| `sample_slicer/detect.py` | `DetectParams`, `Segment`, `find_onsets`, `detect_segments` |
| `sample_slicer/tail.py` | `apply_fade_in`, `natural_tail`, `render_segment` |
| `sample_slicer/slicing.py` | `slice_file` (generický střih → seznam `Slice`), `slice_dir` |
| `sample_slicer/pitch.py` | `Pitch`, `estimate_pitch` |
| `sample_slicer/notes.py` | `note_to_midi`, `midi_to_name`, `expand_truth` |
| `sample_slicer/tuning.py` | `fit_tuning_curve`, `assign_notes`, `Assignment` |
| `sample_slicer/bank.py` | `BankIndex`, `md5_16`, `convert_48k16`, `build_bank`, `write_report` |
| `sample_slicer/cli.py` | `main` s podpříkazy `slice`, `analyze`, `build` |
| `slicer.py` | tenký wrapper: `python slicer.py` = `sample-slicer slice` (zachová staré přepínače) |
| `slicergui/logic.py` | `process_wav_file` volá `sample_slicer.slicing.slice_file` |
| `tools/make_fixtures.py` | vyřeže testovací fixtures z reálných nahrávek |
| `tests/synth.py` | generátor syntetického „piana" pro testy |
| `tests/test_*.py` | testy per modul |
| `tests/fixtures/pitch/` | 53 mono výřezů 0,6 s + `truth.json` |

Datové typy sdílené napříč tasky (definované v tasku, kde vznikají):

```python
# detect.py
@dataclass
class Segment:
    start: int          # index prvního vzorku (včetně pre-rollu)
    onset: int          # index detekovaného nasazení
    end: int            # index konce (exkluzivně) — bod, za kterým se surová data nepoužijí
    peak_db: float      # vrchol obálky v dBFS
    floor_db: float     # lokální šumové dno v dBFS
    slope_db_s: float   # sklon dozvuku (dB/s, záporný = klesá) měřený před koncem
    end_reason: str     # "level" | "artifact" | "max_len" | "next_onset" | "eof"

# pitch.py
@dataclass
class Pitch:
    f0_hz: float
    midi: float         # neceločíselné MIDI (69 + 12*log2(f0/440))
    confidence: float   # skóre vítěze / součet skóre shluků, 0..1
    n_votes: int        # počet k, které hlasovaly pro vítěze
    evidence: int       # počet prominentních parciál 1–4 (0..4)

# tuning.py
@dataclass
class Assignment:
    midi: int | None    # None = odmítnuto
    cents_et: float     # odchylka od temperované noty `midi` (nebo nejbližší)
    cents_curve: float  # odchylka od ladicí křivky
    anchor: bool        # přiřazeno v prvním průchodu
    reason: str         # "anchor" | "curve" | "out_of_tolerance" | "out_of_range" | "low_confidence" | "override"
```

---

### Task 1: Scaffold balíčku a testovacího prostředí

**Files:**
- Create: `pyproject.toml`, `sample_slicer/__init__.py`, `tests/__init__.py`, `tests/test_package.py`, `.gitignore`
- Modify: `requirements.txt` (ponechat, doplnit poznámku, že zdrojem pravdy je pyproject)

**Interfaces:**
- Produces: importovatelný balíček `sample_slicer` s `__version__ = "2.0.0.dev0"`; `.venv/bin/pytest` funguje.

- [ ] **Step 1: pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "sample-slicer"
version = "2.0.0.dev0"
description = "Střih nahrávek na samply, odhad výšky a stavba banky pro ithaca-legacy"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "tqdm>=4.62"]

[project.optional-dependencies]
gui = ["PySide6>=6.4", "platformdirs>=3.0"]
dev = ["pytest>=8"]

[project.scripts]
sample-slicer = "sample_slicer.cli:main"

[tool.setuptools.packages.find]
include = ["sample_slicer*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: `sample_slicer/__init__.py`**

```python
"""sample_slicer — střih nahrávek na samply, odhad výšky, stavba banky."""
__version__ = "2.0.0.dev0"
```

- [ ] **Step 3: `.gitignore`**

```
.venv/
__pycache__/
*.pyc
*.egg-info/
build/
dist/
.pytest_cache/
.DS_Store
```

- [ ] **Step 4: Failing test `tests/test_package.py`**

```python
import sample_slicer

def test_version():
    assert sample_slicer.__version__.startswith("2.")
```

- [ ] **Step 5: Vytvořit venv a nainstalovat**

Run:
```bash
cd ~/Projects/sample-slicer
python3.11 -m venv .venv
.venv/bin/pip install -q -e '.[dev]'
.venv/bin/pytest -q
```
Expected: `1 passed`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml sample_slicer/__init__.py tests/__init__.py tests/test_package.py .gitignore
git commit -m "build: balíček sample_slicer + pytest scaffold"
```

---

### Task 2: WAV I/O (16/24/32 bit, DC offset, mono mix)

**Files:**
- Create: `sample_slicer/io.py`, `tests/test_io.py`

**Interfaces:**
- Produces:
  - `@dataclass WavInfo(sample_rate: int, channels: int, bits: int, frames: int)`
  - `read_wav(path) -> tuple[np.ndarray, WavInfo]` — float32 pole tvaru `(frames, channels)` v rozsahu ⟨-1, 1⟩ (i pro mono je 2D).
  - `write_wav(path, data: np.ndarray, sample_rate: int, bits: int)` — `data` float32 `(frames, channels)`, ořízne do ⟨-1, 1⟩, zapíše PCM int.
  - `remove_dc(data) -> np.ndarray` — odečte průměr per kanál.
  - `to_mono(data) -> np.ndarray` — 1D `mean(axis=1)`.
  - `class UnsupportedWav(Exception)`.

- [ ] **Step 1: Failing testy**

```python
# tests/test_io.py
import numpy as np, pytest
from sample_slicer.io import read_wav, write_wav, remove_dc, to_mono, UnsupportedWav, WavInfo

@pytest.mark.parametrize("bits", [16, 24, 32])
def test_roundtrip_bits(tmp_path, bits):
    sr = 48000
    t = np.arange(4800) / sr
    data = np.stack([0.5*np.sin(2*np.pi*440*t), -0.25*np.sin(2*np.pi*220*t)], axis=1).astype(np.float32)
    p = tmp_path / f"x{bits}.wav"
    write_wav(p, data, sr, bits)
    back, info = read_wav(p)
    assert info == WavInfo(sample_rate=sr, channels=2, bits=bits, frames=4800)
    assert back.shape == (4800, 2)
    assert np.max(np.abs(back - data)) < 2.0 / (2 ** (bits - 1)) + 1e-6

def test_roundtrip_24bit_bit_exact(tmp_path):
    # 24bit hodnoty přesně reprezentovatelné → po round-tripu identické bajty
    sr = 96000
    q = 2 ** 23
    ints = np.array([[-q, q - 1], [0, 1], [-1, 12345]], dtype=np.int64)
    data = (ints / q).astype(np.float32)
    p = tmp_path / "exact.wav"
    write_wav(p, data, sr, 24)
    back, info = read_wav(p)
    assert info.bits == 24 and info.channels == 2
    assert np.array_equal(np.round(back * q).astype(np.int64), ints)

def test_mono_is_2d(tmp_path):
    data = np.zeros((100, 1), dtype=np.float32)
    p = tmp_path / "m.wav"
    write_wav(p, data, 44100, 16)
    back, info = read_wav(p)
    assert back.shape == (100, 1) and info.channels == 1

def test_remove_dc_and_mono():
    data = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]], dtype=np.float32)
    nodc = remove_dc(data)
    assert np.allclose(nodc.mean(axis=0), 0)
    assert np.allclose(to_mono(data), [0.5, 0.75, 1.0])

def test_float_wav_rejected(tmp_path):
    import struct
    p = tmp_path / "f.wav"
    pcm = np.zeros(10, dtype=np.float32).tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, 48000, 48000*4, 4, 32) + b"data" + struct.pack("<I", len(pcm))
    p.write_bytes(hdr + pcm)
    with pytest.raises(UnsupportedWav):
        read_wav(p)
```

- [ ] **Step 2: Ověřit, že selhávají**

Run: `.venv/bin/pytest tests/test_io.py -q`
Expected: FAIL (ImportError `sample_slicer.io`).

- [ ] **Step 3: Implementace `sample_slicer/io.py`**

```python
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
    # `wave` sám float WAV odmítne nebo mylně přijme; ověříme fmt tag ručně.
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
```

- [ ] **Step 4: Testy projdou**

Run: `.venv/bin/pytest tests/test_io.py -q`
Expected: `7 passed` (parametrize dává 3 testy + 4 další).

- [ ] **Step 5: Commit**

```bash
git add sample_slicer/io.py tests/test_io.py
git commit -m "feat(io): čtení/zápis WAV 16/24/32 bit, DC offset, mono mix"
```

---

### Task 3: Obálka, lokální dno, sklon dozvuku

**Files:**
- Create: `sample_slicer/envelope.py`, `tests/test_envelope.py`

**Interfaces:**
- Produces:
  - `rms_envelope_db(mono: np.ndarray, sr: int, hop_ms: float = 5.0) -> np.ndarray` — dB per rámec, podlaha -120 dB.
  - `smooth_db(env: np.ndarray, frames: int) -> np.ndarray` — klouzavý medián lichého okna (posun 0).
  - `local_floor_db(env, i_onset: int, hop_s: float, window_s: float = 2.0) -> float` — 5. percentil `env[max(0, i_onset - window):i_onset]`; není-li nic, minimum celé obálky.
  - `decay_slope_db_s(env, i_end: int, hop_s: float, window_s: float = 2.0) -> float` — směrnice lineární regrese (dB/s) přes `env[i_end - window : i_end]`.
  - `hop_frames(sr, hop_ms) -> int`.

- [ ] **Step 1: Failing testy**

```python
# tests/test_envelope.py
import numpy as np
from sample_slicer.envelope import rms_envelope_db, smooth_db, local_floor_db, decay_slope_db_s, hop_frames

SR = 48000

def test_envelope_of_constant_sine():
    t = np.arange(SR) / SR
    x = 0.5 * np.sin(2 * np.pi * 1000 * t)
    env = rms_envelope_db(x, SR, hop_ms=10)
    assert env.shape[0] == 100
    # RMS sinusu 0.5 = 0.3536 → -9.03 dB
    assert np.allclose(env[5:-5], -9.03, atol=0.2)

def test_envelope_silence_floor():
    env = rms_envelope_db(np.zeros(SR), SR)
    assert np.all(env == -120.0)

def test_smooth_median_removes_spike():
    env = np.full(50, -40.0); env[25] = 0.0
    assert smooth_db(env, 5)[25] == -40.0

def test_local_floor():
    hop = 0.005
    env = np.full(1000, -70.0); env[300:400] = -20.0   # tón v 1.5–2.0 s
    # před nasazením v 3.0 s (index 600) je okno 2 s = indexy 200..600, 5. percentil ~ -70
    assert local_floor_db(env, 600, hop) == -70.0
    assert local_floor_db(env, 0, hop) == -70.0

def test_decay_slope():
    hop = 0.005
    t = np.arange(2000) * hop
    env = -20 - 3.0 * t   # -3 dB/s
    assert abs(decay_slope_db_s(env, 2000, hop) - (-3.0)) < 1e-6

def test_hop_frames():
    assert hop_frames(96000, 5.0) == 480
```

- [ ] **Step 2: Ověřit selhání** — `.venv/bin/pytest tests/test_envelope.py -q` → ImportError.

- [ ] **Step 3: Implementace `sample_slicer/envelope.py`**

```python
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
```

- [ ] **Step 4: Testy projdou** — `.venv/bin/pytest tests/test_envelope.py -q` → `6 passed`.

- [ ] **Step 5: Commit**

```bash
git add sample_slicer/envelope.py tests/test_envelope.py
git commit -m "feat(envelope): RMS obálka, lokální dno, sklon dozvuku"
```

---

### Task 4: Syntetický generátor „piana" pro testy

**Files:**
- Create: `tests/synth.py`, `tests/test_synth.py`

**Interfaces:**
- Produces:
  - `piano_note(midi: float, sr: int, dur_s: float, peak: float = 0.5, decay_db_s: float = -6.0, inharmonicity: float = 0.0, n_partials: int = 8, attack_ms: float = 3.0, seed: int = 0) -> np.ndarray` — 1D float32; parciály `h*f0*sqrt(1+B*h^2)` s amplitudou `1/h`, exponenciální dozvuk, krátký attack.
  - `thump(sr, dur_ms=30.0, peak=0.05, seed=0) -> np.ndarray` — širokopásmový šum s rychlým dozvukem (pád kladívka).
  - `place(canvas: np.ndarray, x: np.ndarray, at_s: float, sr: int) -> None` — přičte `x` do `canvas` od času `at_s`.
  - `noise_floor(n, sr, db=-80.0, seed=0) -> np.ndarray`.

- [ ] **Step 1: Failing test `tests/test_synth.py`**

```python
import numpy as np
from tests.synth import piano_note, thump, place, noise_floor

def test_piano_note_shape_and_peak():
    x = piano_note(69, 48000, 1.0, peak=0.5)
    assert x.shape == (48000,) and x.dtype == np.float32
    assert 0.45 < np.abs(x).max() <= 0.5

def test_piano_note_decays():
    x = piano_note(69, 48000, 2.0, decay_db_s=-20.0)
    a = np.sqrt(np.mean(x[4800:9600] ** 2)); b = np.sqrt(np.mean(x[48000:52800] ** 2))
    assert 20 * np.log10(b / a) < -12   # za ~0.9 s aspoň 12 dB dolů

def test_place_and_floor():
    c = noise_floor(48000, 48000, db=-80)
    place(c, np.ones(10, dtype=np.float32), 0.5, 48000)
    assert c[24000] > 0.9 and abs(c[100]) < 1e-3
    assert len(thump(48000)) == 1440
```

- [ ] **Step 2: Ověřit selhání** — `.venv/bin/pytest tests/test_synth.py -q` → ImportError.

- [ ] **Step 3: Implementace `tests/synth.py`**

```python
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
```

- [ ] **Step 4: Testy projdou** — `.venv/bin/pytest tests/test_synth.py -q` → `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add tests/synth.py tests/test_synth.py
git commit -m "test: syntetický generátor pianového tónu, thumpu a šumu"
```

---

### Task 5: Detekce segmentů (nasazení, dělení, konec, artefakt, kliky)

**Files:**
- Create: `sample_slicer/detect.py`, `tests/test_detect.py`

**Interfaces:**
- Consumes: `envelope.rms_envelope_db`, `smooth_db`, `local_floor_db`, `decay_slope_db_s`, `hop_frames`.
- Produces:
  - `@dataclass DetectParams` s poli (výchozí hodnoty = Global Constraints): `hop_ms=5.0, onset_rise_db=20.0, onset_rise_ms=30.0, preroll_ms=5.0, split_rise_db=12.0, split_peak_within_db=20.0, split_min_len_s=0.5, click_below_peak_db=25.0, end_level_db=-60.0, artifact_after_s=1.0, artifact_window_s=2.0, artifact_rise_db=8.0, max_len_s=30.0, smooth_ms=100.0, min_len_s=0.3`.
  - `@dataclass Segment` (viz Mapa souborů).
  - `find_onsets(env: np.ndarray, hop_s: float, p: DetectParams) -> list[int]` — indexy rámců nasazení (po zpětném hledání minima), už sloučené/rozdělené podle pravidla slitých tónů.
  - `detect_segments(mono: np.ndarray, sr: int, p: DetectParams = DetectParams()) -> tuple[list[Segment], list[Segment]]` — `(accepted, rejected_clicks)`.

- [ ] **Step 1: Failing testy**

```python
# tests/test_detect.py
import numpy as np
from sample_slicer.detect import detect_segments, DetectParams
from tests.synth import piano_note, thump, place, noise_floor

SR = 48000

def make_canvas(sec):
    return noise_floor(int(sec * SR), SR, db=-85)

def test_two_notes_onsets_within_5ms():
    c = make_canvas(8)
    place(c, piano_note(60, SR, 3.0, decay_db_s=-15), 1.0, SR)
    place(c, piano_note(64, SR, 3.0, decay_db_s=-15), 5.0, SR)
    segs, rej = detect_segments(c, SR)
    assert len(segs) == 2 and rej == []
    assert abs(segs[0].onset / SR - 1.0) < 0.005
    assert abs(segs[1].onset / SR - 5.0) < 0.005
    assert 0 <= segs[0].onset - segs[0].start <= int(0.02 * SR)   # pre-roll ≤ 20 ms

def test_merged_notes_are_split():
    c = make_canvas(6)
    place(c, piano_note(60, SR, 4.0, decay_db_s=-10), 1.0, SR)
    place(c, piano_note(67, SR, 3.0, decay_db_s=-10), 2.0, SR)   # druhý tón do dozvuku prvního
    segs, _ = detect_segments(c, SR)
    assert len(segs) == 2
    assert segs[0].end_reason == "next_onset" and segs[0].end <= segs[1].start
    assert abs(segs[1].onset / SR - 2.0) < 0.005

def test_end_at_level():
    c = make_canvas(12)
    place(c, piano_note(60, SR, 10.0, peak=0.5, decay_db_s=-10), 1.0, SR)   # -6 dBFS peak, -60 dB za ~5.4 s
    segs, _ = detect_segments(c, SR, DetectParams(end_level_db=-40.0))
    assert len(segs) == 1 and segs[0].end_reason == "level"
    assert 3.0 < (segs[0].end - segs[0].onset) / SR < 5.0
    assert segs[0].slope_db_s < -5

def test_release_artifact_cuts_before_thump():
    c = make_canvas(10)
    place(c, piano_note(60, SR, 8.0, peak=0.5, decay_db_s=-6), 1.0, SR)
    place(c, thump(SR, peak=0.2), 4.0, SR)                       # pád kladívka ve 4.0 s
    segs, _ = detect_segments(c, SR, DetectParams(end_level_db=-90.0))
    assert len(segs) == 1 and segs[0].end_reason == "artifact"
    assert 3.7 < segs[0].end / SR <= 4.0

def test_max_len():
    c = make_canvas(6)
    place(c, piano_note(60, SR, 5.0, decay_db_s=-1), 0.5, SR)
    segs, _ = detect_segments(c, SR, DetectParams(max_len_s=2.0, end_level_db=-90.0))
    assert segs[0].end_reason == "max_len" and abs((segs[0].end - segs[0].onset) / SR - 2.0) < 0.02

def test_clicks_rejected():
    c = make_canvas(6)
    place(c, piano_note(60, SR, 2.0, peak=0.5), 1.0, SR)
    place(c, thump(SR, peak=0.005), 4.0, SR)                      # 40 dB pod tónem
    segs, rej = detect_segments(c, SR)
    assert len(segs) == 1 and len(rej) == 1
    assert rej[0].peak_db < segs[0].peak_db - 25
```

- [ ] **Step 2: Ověřit selhání** — `.venv/bin/pytest tests/test_detect.py -q` → ImportError.

- [ ] **Step 3: Implementace `sample_slicer/detect.py`**

```python
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


def _backtrack_min(env: np.ndarray, i: int, max_back: int) -> int:
    """Od rámce i jdi zpět, dokud obálka klesá (hledáme lokální minimum před nasazením)."""
    j = i
    while j > 0 and i - j < max_back and env[j - 1] <= env[j]:
        j -= 1
    return j


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
                onsets.append(_backtrack_min(env, i - rise_frames, rise_frames * 4))
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
                    onsets.append(_backtrack_min(env, i - rise_frames, rise_frames * 4))
                    active_peak = new_peak
                    i += rise_frames
                    continue
            # konec aktivního úseku = obálka spadla k dnu; pak se zase hledá nasazení
            floor = local_floor_db(env, onsets[-1], hop_s)
            if env[i] < floor + 6.0:
                active_peak = None
        i += 1
    return onsets


def _segment_end(env_s: np.ndarray, i_onset: int, i_limit: int, hop_s: float, p: DetectParams):
    """Vrátí (i_end, reason, slope). env_s je vyhlazená obálka; i_limit = začátek dalšího úderu nebo len."""
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
            if env_s[i] - predicted > p.artifact_rise_db:
                back = int(round(0.1 / hop_s))
                j = i - back + int(np.argmin(env_s[i - back: i])) if back > 0 else i
                return j, "artifact", float(slope)
        i += 1
    reason = "next_onset" if i_limit < len(env_s) and i_max == i_limit else ("max_len" if i_max < i_limit else "eof")
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
        i_end, reason, slope = _segment_end(env_s, i_on, i_limit, hop_s, p)
        onset = i_on * hop
        start = max(0, onset - preroll)
        end = min(len(mono), i_end * hop)
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
```

- [ ] **Step 4: Testy projdou** — `.venv/bin/pytest tests/test_detect.py -q` → `6 passed`. Pokud některý test neprojde kvůli prahům syntetiky (ne kvůli logice), uprav parametry syntetiky v testu (peak, decay), ne výchozí hodnoty `DetectParams` — ty jsou dané specem.

- [ ] **Step 5: Commit**

```bash
git add sample_slicer/detect.py tests/test_detect.py
git commit -m "feat(detect): nasazení, dělení slitých tónů, konec, artefakt uvolnění, kliky"
```

---

### Task 6: Přirozený dozvuk do nuly a fade-in

**Files:**
- Create: `sample_slicer/tail.py`, `tests/test_tail.py`

**Interfaces:**
- Consumes: `detect.Segment`.
- Produces:
  - `apply_fade_in(data: np.ndarray, sr: int, ms: float = 2.0) -> np.ndarray` — in-place lineární fade, vrací `data`.
  - `natural_tail(data: np.ndarray, sr: int, slope_db_s: float, tail_s: float = 2.0, floor_db: float = -96.0, final_ms: float = 10.0) -> np.ndarray` — na **posledních `tail_s`** vstupu aplikuje exponenciální útlum se strmostí `max(|slope|, floor/tail_s)` dB/s, posledních `final_ms` lineárně do nuly; vrací nové pole stejné délky.
  - `render_segment(audio: np.ndarray, sr: int, seg: Segment, tail_s: float = 2.0, fade_in_ms: float = 2.0) -> np.ndarray` — vyřízne `audio[seg.start : seg.end + tail_s*sr]` (2D, `(frames, ch)`), fade-in, `natural_tail` na posledních `tail_s`; když za `seg.end` chybí data (EOF), doplní nulami.

- [ ] **Step 1: Failing testy**

```python
# tests/test_tail.py
import numpy as np
from sample_slicer.tail import apply_fade_in, natural_tail, render_segment
from sample_slicer.detect import Segment
from sample_slicer.envelope import rms_envelope_db

SR = 48000

def test_fade_in_starts_at_zero():
    x = np.ones((SR, 1), dtype=np.float32)
    y = apply_fade_in(x, SR, ms=2.0)
    assert y[0, 0] == 0.0 and y[95, 0] < 1.0 and y[96, 0] == 1.0

def test_natural_tail_monotone_and_ends_zero():
    x = np.ones((4 * SR, 1), dtype=np.float32)
    y = natural_tail(x, SR, slope_db_s=-3.0, tail_s=2.0)
    assert np.array_equal(y[: 2 * SR], x[: 2 * SR])          # před ocasem beze změny
    env = rms_envelope_db(y[:, 0], SR, hop_ms=50)
    tail_env = env[40:]                                        # posledních 2 s
    assert np.all(np.diff(tail_env) <= 1e-6)                   # monotónně klesá
    assert y[-1, 0] == 0.0
    # strmost ≥ požadovaná: za 2 s musí dojít aspoň na -96 dB
    assert env[-2] < -80

def test_natural_tail_uses_measured_slope_when_steeper():
    x = np.ones((3 * SR, 1), dtype=np.float32)
    y = natural_tail(x, SR, slope_db_s=-100.0, tail_s=2.0)     # naměřených -100 dB/s je strmější než 48 dB/s
    env = rms_envelope_db(y[:, 0], SR, hop_ms=50)
    assert env[len(env) - 40 + 10] < -90                       # po 0.5 s ocasu už -100 dB

def test_render_segment_pads_at_eof():
    audio = np.ones((SR, 2), dtype=np.float32)
    seg = Segment(start=0, onset=100, end=SR - 100, peak_db=0, floor_db=-80, slope_db_s=-3, end_reason="eof")
    out = render_segment(audio, SR, seg, tail_s=1.0)
    assert out.shape == (SR - 100 + SR, 2)
    assert out[-1, 0] == 0.0 and out[0, 0] == 0.0
```

- [ ] **Step 2: Ověřit selhání** — `.venv/bin/pytest tests/test_tail.py -q` → ImportError.

- [ ] **Step 3: Implementace `sample_slicer/tail.py`**

```python
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
    n_tail = int(round(tail_s * sr))
    stop = seg.end + n_tail
    piece = audio[seg.start: min(stop, len(audio))].astype(np.float32, copy=True)
    if stop > len(audio):
        piece = np.concatenate([piece, np.zeros((stop - len(audio), audio.shape[1]), dtype=np.float32)])
    apply_fade_in(piece, sr, fade_in_ms)
    return natural_tail(piece, sr, seg.slope_db_s, tail_s)
```

- [ ] **Step 4: Testy projdou** — `.venv/bin/pytest tests/test_tail.py -q` → `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add sample_slicer/tail.py tests/test_tail.py
git commit -m "feat(tail): přirozený dozvuk do nuly, fade-in"
```

---

### Task 7: Generický střih (`slice_file`), CLI `slice`, přepojení `slicer.py` a GUI

**Files:**
- Create: `sample_slicer/slicing.py`, `sample_slicer/cli.py`, `tests/test_slicing.py`
- Modify: `slicer.py` (celý nahradit wrapperem), `slicergui/logic.py:362-509` (`process_wav_file` volá `slice_file`; funkce `apply_fade`, `detect_segments`, `trim_silence`, `load_audio_data`, `save_audio_segment` z logic.py smazat), `README.md` (sekce Použití)

**Interfaces:**
- Consumes: `io.read_wav/write_wav/remove_dc/to_mono`, `detect.detect_segments/DetectParams/Segment`, `tail.render_segment`.
- Produces:
  - `@dataclass Slice(index: int, segment: Segment, audio: np.ndarray, sr: int, bits: int, channels: int, source: Path)`.
  - `slice_file(path, params: DetectParams = DetectParams(), tail_s: float = 2.0, fade_in_ms: float = 2.0) -> tuple[list[Slice], list[Segment]]` — `(slices, rejected_clicks)`; audio je stereo/mono podle vstupu.
  - `slice_dir(in_dir, out_dir, params, tail_s, fade_in_ms, log=print) -> int` — zapíše `<stem>_slice_NNN_start_<ms>ms_dur_<ms>ms.wav`, vrací počet.
  - `cli.main(argv=None) -> int` s podpříkazem `slice` (přepínače `--threshold-db` → mapuje na `end_level_db`? NE: `slice` má vlastní přepínače `--end-level-db`, `--max-len-s`, `--tail-s`, `--preroll-ms`; staré přepínače `slicer.py` (`--threshold_db`, `--min_length`, …) wrapper přijme a vypíše upozornění, že se ignorují, kromě `--input-dir/--output-dir`).
  - `add_detect_args(parser)` a `detect_params_from_args(args) -> DetectParams` (sdílené pro `analyze`/`build` v dalších taskách).

- [ ] **Step 1: Failing testy**

```python
# tests/test_slicing.py
import numpy as np
from sample_slicer.io import write_wav, read_wav
from sample_slicer.slicing import slice_file, slice_dir
from sample_slicer.cli import main
from tests.synth import piano_note, place, noise_floor

SR = 48000

def make_file(path, notes=(60, 64), bits=24):
    c = noise_floor(int(10 * SR), SR, db=-85)
    for i, m in enumerate(notes):
        place(c, piano_note(m, SR, 3.0, decay_db_s=-15), 1.0 + 4.0 * i, SR)
    write_wav(path, np.stack([c, 0.8 * c], axis=1), SR, bits)

def test_slice_file_returns_stereo_slices(tmp_path):
    p = tmp_path / "in.wav"; make_file(p)
    slices, rejected = slice_file(p)
    assert len(slices) == 2 and rejected == []
    s = slices[0]
    assert s.audio.ndim == 2 and s.audio.shape[1] == 2 and s.bits == 24 and s.sr == SR
    assert s.audio[-1, 0] == 0.0                     # ocas do nuly
    assert s.index == 0 and s.source == p

def test_slice_dir_writes_named_files(tmp_path):
    src = tmp_path / "src"; src.mkdir(); make_file(src / "rec.wav")
    out = tmp_path / "out"
    n = slice_dir(src, out)
    files = sorted(out.glob("rec_slice_*.wav"))
    assert n == 2 and len(files) == 2
    assert files[0].name.startswith("rec_slice_001_start_")
    data, info = read_wav(files[0])
    assert info.bits == 24 and info.channels == 2

def test_cli_slice(tmp_path):
    src = tmp_path / "src"; src.mkdir(); make_file(src / "rec.wav")
    out = tmp_path / "out"
    assert main(["slice", str(src), str(out)]) == 0
    assert len(list(out.glob("*.wav"))) == 2
```

- [ ] **Step 2: Ověřit selhání** — `.venv/bin/pytest tests/test_slicing.py -q` → ImportError.

- [ ] **Step 3: Implementace `sample_slicer/slicing.py`**

```python
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
```

- [ ] **Step 4: Implementace `sample_slicer/cli.py` (zatím jen `slice`)**

```python
"""CLI: sample-slicer slice | analyze | build."""
from __future__ import annotations
import argparse, sys
from dataclasses import fields
from .detect import DetectParams
from . import slicing


def add_detect_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("detekce (výchozí hodnoty ze specu)")
    for f in fields(DetectParams):
        g.add_argument("--" + f.name.replace("_", "-"), type=float, default=None, metavar="X",
                       help=f"výchozí {getattr(DetectParams(), f.name)}")
    g.add_argument("--tail-s", type=float, default=2.0, help="délka přirozeného dozvuku (s)")
    g.add_argument("--fade-in-ms", type=float, default=2.0)


def detect_params_from_args(args) -> DetectParams:
    kw = {f.name: getattr(args, f.name) for f in fields(DetectParams) if getattr(args, f.name) is not None}
    return DetectParams(**kw)


def cmd_slice(args) -> int:
    n = slicing.slice_dir(args.in_dir, args.out_dir, detect_params_from_args(args), args.tail_s, args.fade_in_ms)
    print(f"Uloženo {n} úderů do {args.out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sample-slicer")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("slice", help="generický střih: adresář WAV → ořezané údery")
    s.add_argument("in_dir"); s.add_argument("out_dir")
    add_detect_args(s)
    s.set_defaults(func=cmd_slice)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: `slicer.py` jako wrapper**

Nahraď celý obsah `slicer.py`:

```python
#!/usr/bin/env python3
"""Zpětně kompatibilní vstup: `python slicer.py --input-dir A --output-dir B`.

Logika žije v balíčku sample_slicer (viz `sample-slicer slice --help`). Staré
přepínače detekce (--threshold_db, --min_length, …) už nemají význam — nový
algoritmus pracuje s lokálním šumovým dnem a přirozeným dozvukem — a jsou
ignorovány s varováním.
"""
import argparse, sys
from sample_slicer.cli import main

IGNORED = ["--threshold_db", "--min_length", "--min_length_after_trim", "--trim_threshold_offset",
           "--fade_ms", "--no_fades", "--resume", "--preview", "--log_level"]

if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    for opt in IGNORED:
        if opt in ("--no_fades", "--resume", "--preview"):
            p.add_argument(opt, action="store_true")
        else:
            p.add_argument(opt, default=None)
    a, rest = p.parse_known_args()
    used = [o for o in IGNORED if getattr(a, o.lstrip("-")) not in (None, False)]
    if used:
        print(f"UPOZORNĚNÍ: přepínače {', '.join(used)} se v nové verzi ignorují", file=sys.stderr)
    sys.exit(main(["slice", a.input_dir, a.output_dir] + rest))
```

- [ ] **Step 6: GUI `slicergui/logic.py`**

Smaž z `logic.py` funkce `apply_fade`, `detect_segments`, `trim_silence`, `get_audio_info`, `load_audio_data`, `save_audio_segment` (řádky 71–361) a `process_wav_file` přepiš tak, aby zachovala signaturu (worker ji volá s `threshold_db`, `min_length`, `min_length_after_trim`, `trim_threshold_offset`, `apply_fades`, `fade_in_ms`, `fade_out_percent`, `overwrite`, `stats`, `progress_callback`, `log_callback`) a volala nový střih:

```python
from pathlib import Path
from sample_slicer.slicing import slice_file
from sample_slicer.io import write_wav, UnsupportedWav
from sample_slicer.detect import DetectParams


def process_wav_file(input_path, output_dir, threshold_db=-45.0, min_length=3.0,
                     min_length_after_trim=0.5, trim_threshold_offset=10.0, apply_fades=True,
                     fade_in_ms=5.0, fade_out_percent=20.0, overwrite=True, stats=None,
                     progress_callback=None, log_callback=None) -> bool:
    """Střih jednoho souboru přes sample_slicer. Staré parametry prahů se mapují takto:
    threshold_db → end_level_db (konec dozvuku), fade_in_ms → fade-in; ostatní se ignorují."""
    def log(msg, level="INFO"):
        if log_callback:
            log_callback(msg, level)
    params = DetectParams(end_level_db=float(threshold_db))
    try:
        slices, rejected = slice_file(input_path, params, tail_s=2.0,
                                      fade_in_ms=float(fade_in_ms) if apply_fades else 0.0)
    except UnsupportedWav as e:
        log(str(e), "ERROR")
        if stats: stats.files_failed += 1
        return False
    src = Path(input_path)
    for i, s in enumerate(slices):
        start_ms = int(s.segment.start / s.sr * 1000); dur_ms = int(len(s.audio) / s.sr * 1000)
        out = Path(output_dir) / f"{src.stem}_slice_{s.index + 1:03d}_start_{start_ms}ms_dur_{dur_ms}ms.wav"
        if out.exists() and not overwrite:
            if stats: stats.segments_skipped += 1
            continue
        write_wav(out, s.audio, s.sr, s.bits)
        if stats:
            stats.segments_created += 1
            stats.total_output_duration += len(s.audio) / s.sr
        if progress_callback:
            progress_callback(100.0 * (i + 1) / max(1, len(slices)))
    log(f"{src.name}: {len(slices)} úderů, {len(rejected)} kliků zahozeno")
    if stats:
        stats.files_processed += 1
    return True
```

Ponech v `logic.py` `ProcessingStats` a `validate_parameters`. Ověř import: `.venv/bin/python -c "import slicergui.logic"` (PySide6 není potřeba pro logic.py; pokud logic.py importuje Qt, přesuň tyto importy do gui.py).

- [ ] **Step 7: README — sekce Použití**

V `README.md` nahraď sekce „Použití", „Parametry" a „Algoritmus zpracování" krátkým textem: instalace (`python3.11 -m venv .venv && .venv/bin/pip install -e .`), `sample-slicer slice <in> <out>`, odkaz na spec (`docs/superpowers/specs/2026-09-21-bank-pipeline-design.md`) pro popis algoritmu a poznámka, že `python slicer.py --input-dir/--output-dir` dál funguje. Sekce `analyze`/`build` se doplní v Tasku 13.

- [ ] **Step 8: Testy projdou**

Run: `.venv/bin/pytest -q`
Expected: všechny testy zelené (`test_slicing.py`: `3 passed`).

- [ ] **Step 9: Commit**

```bash
git add sample_slicer/slicing.py sample_slicer/cli.py tests/test_slicing.py slicer.py slicergui/logic.py README.md
git commit -m "feat(slicing): slice_file + CLI slice; slicer.py a GUI jako tenká vrstva"
```

---

### Task 8: Odhad výšky (`pitch.py`)

**Files:**
- Create: `sample_slicer/pitch.py`, `tests/test_pitch.py`

**Interfaces:**
- Produces:
  - `@dataclass Pitch` (viz Mapa souborů) + property `cents_et` = odchylka od nejbližší celé noty.
  - `estimate_pitch(x: np.ndarray, sr: int, offset_s: float = 0.04, win_nom_s: float = 0.25, ks=(0.25, 0.5, 1, 2, 4, 8, 16), fmin_nom=100.0, fmax_nom=1600.0, min_peak=0.6, cluster_cents=50.0, prominence_db=12.0, min_evidence=2, refine_cents=150.0) -> Pitch | None` — `x` je 1D mono úsek **začínající nasazením**; `None`, když žádné k nedá vrchol.
  - `midi_from_hz(f) -> float`, `hz_from_midi(m) -> float`.

- [ ] **Step 1: Failing testy**

```python
# tests/test_pitch.py
import numpy as np, pytest
from sample_slicer.pitch import estimate_pitch, midi_from_hz, hz_from_midi
from tests.synth import piano_note, thump, place, noise_floor

SR = 96000

def note_with_context(midi, dur=1.0, **kw):
    x = noise_floor(int(dur * SR), SR, db=-80)
    x += piano_note(midi, SR, dur, **kw)
    return x

@pytest.mark.parametrize("midi", [21, 24, 33, 45, 60, 72, 84, 96, 105, 108])
def test_synthetic_notes_across_keyboard(midi):
    x = note_with_context(midi, inharmonicity=0.0005 if midi > 60 else 0.0001, decay_db_s=-8 if midi < 60 else -30)
    p = estimate_pitch(x, SR)
    assert p is not None
    assert abs(p.midi - midi) < 0.3, (p, midi)
    assert p.evidence >= 2

def test_weak_fundamental_bass():
    # A0 s fundamentálem 30 dB pod parciálami 2–6 (jako u reálného basu)
    dur = 1.5
    t = np.arange(int(dur * SR)) / SR
    f0 = hz_from_midi(21)
    x = np.zeros_like(t)
    for h, amp in [(1, 0.03), (2, 0.6), (3, 0.8), (4, 1.0), (5, 0.7), (6, 0.5)]:
        x += amp * np.exp(-2 * t) * np.sin(2 * np.pi * h * f0 * np.sqrt(1 + 1e-4 * h * h) * t)
    x = (0.4 * x / np.abs(x).max()).astype(np.float32) + noise_floor(len(t), SR, db=-80)
    p = estimate_pitch(x, SR)
    assert abs(p.midi - 21) < 0.3, p

def test_thump_does_not_win_on_short_treble():
    x = note_with_context(100, dur=0.7, decay_db_s=-40, inharmonicity=0.001)
    place(x, thump(SR, peak=0.3), 0.0, SR)       # silný úder kladívka na začátku
    p = estimate_pitch(x, SR)
    assert abs(p.midi - 100) < 0.3, p

def test_conversions():
    assert abs(midi_from_hz(440) - 69) < 1e-9 and abs(hz_from_midi(21) - 27.5) < 1e-9

def test_silence_returns_none():
    assert estimate_pitch(np.zeros(SR, dtype=np.float32), SR) is None
```

- [ ] **Step 2: Ověřit selhání** — `.venv/bin/pytest tests/test_pitch.py -q` → ImportError.

- [ ] **Step 3: Implementace `sample_slicer/pitch.py`**

```python
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
```

- [ ] **Step 4: Testy projdou** — `.venv/bin/pytest tests/test_pitch.py -q` → `14 passed`. Když syntetický test selže o oktávu u extrémů (21, 108), zkontroluj nejdřív syntetiku (délka ≥ 0,7 s, peak, decay); algoritmus měň jen, když selhává i reálný fixture test v Tasku 9.

- [ ] **Step 5: Commit**

```bash
git add sample_slicer/pitch.py tests/test_pitch.py
git commit -m "feat(pitch): ACF s normalizací pásma, hlasování, spektrální důkaz, doladění"
```

---

### Task 9: Fixtures z reálných nahrávek + regresní test pitch

**Files:**
- Create: `sample_slicer/notes.py`, `tools/make_fixtures.py`, `tests/fixtures/pitch/truth.json`, `tests/fixtures/pitch/*.wav` (53 souborů), `tests/test_notes.py`, `tests/test_pitch_fixtures.py`

**Interfaces:**
- Produces (`notes.py`):
  - `note_to_midi("A0") -> 21`, `("C#4") -> 61`, `("Db4") -> 61`, `("C4") -> 60`.
  - `midi_to_name(60) -> "C4"`.
  - `expand_truth(entry: dict, count: int) -> list[int]` — `{"start": "A0", "pattern": "chromatic"}` → `[21, 22, …]` délky `count`; `"major"` → intervaly `[2,2,1,2,2,2,1]`; `{"pattern": "list", "notes": ["C4", "E4"]}` → výčet (count se ignoruje, vrátí celý výčet).
- Fixtures: `tests/fixtures/pitch/<stem>_<NN>.wav` mono 24 bit 96 kHz, 0,6 s od `Segment.onset`; `truth.json` = `{"<soubor>.wav": midi, ...}`.

- [ ] **Step 1: Failing testy `tests/test_notes.py`**

```python
import pytest
from sample_slicer.notes import note_to_midi, midi_to_name, expand_truth

def test_note_names():
    assert note_to_midi("A0") == 21 and note_to_midi("C4") == 60 and note_to_midi("C#4") == 61
    assert note_to_midi("Db4") == 61 and note_to_midi("c8") == 108
    assert midi_to_name(60) == "C4" and midi_to_name(21) == "A0" and midi_to_name(61) == "C#4"

def test_expand_chromatic_and_major():
    assert expand_truth({"start": "A0", "pattern": "chromatic"}, 4) == [21, 22, 23, 24]
    assert expand_truth({"start": "C4", "pattern": "major"}, 8) == [60, 62, 64, 65, 67, 69, 71, 72]
    assert expand_truth({"pattern": "list", "notes": ["C4", "E4"]}, 99) == [60, 64]

def test_bad_pattern():
    with pytest.raises(ValueError):
        expand_truth({"start": "C4", "pattern": "minor"}, 3)
```

- [ ] **Step 2: Implementace `sample_slicer/notes.py`**

```python
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
```

Run: `.venv/bin/pytest tests/test_notes.py -q` → `3 passed`.

- [ ] **Step 3: `tools/make_fixtures.py`**

```python
#!/usr/bin/env python3
"""Vyřeže testovací fixtures pro pitch z reálných nahrávek.

Použití: tools/make_fixtures.py <raw-dir> <truth.json> <out-dir>
truth.json: {"260917_0180.wav": {"start": "A0", "pattern": "chromatic"}, ...}
Pro každý soubor: detekce úderů, kontrola, že počet přijatých úderů sedí s
očekáváním (pravda musí mít pevný počet → u chromatic/major se počet bere z
detekce a vypíše se k ruční kontrole), zápis 0,6 s mono 24 bit od nasazení.
"""
import json, sys
from pathlib import Path
import numpy as np
from sample_slicer.io import read_wav, write_wav, remove_dc, to_mono
from sample_slicer.detect import detect_segments
from sample_slicer.notes import expand_truth, midi_to_name

raw, truth_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
truth = json.loads(truth_path.read_text())
result = {}
for name, entry in truth.items():
    data, info = read_wav(raw / name)
    mono = to_mono(remove_dc(data))
    accepted, rejected = detect_segments(mono, info.sample_rate)
    midis = expand_truth(entry, len(accepted))
    print(f"{name}: {len(accepted)} úderů ({len(rejected)} kliků) → {midi_to_name(midis[0])}..{midi_to_name(midis[-1])}")
    for i, (seg, midi) in enumerate(zip(accepted, midis)):
        clip = mono[seg.onset: seg.onset + int(0.6 * info.sample_rate)].astype(np.float32)[:, None]
        fn = f"{Path(name).stem}_{i:02d}.wav"
        write_wav(out / fn, clip, info.sample_rate, 24)
        result[fn] = midi
(out / "truth.json").write_text(json.dumps(result, indent=1))
print(f"zapsáno {len(result)} fixtures do {out}")
```

- [ ] **Step 4: Vygenerovat fixtures**

Vytvoř `/Users/j/SoundBanks/Ithaca/raw-samples-legacy-01/truth.json`:

```json
{
  "260917_0180.wav": {"start": "A0", "pattern": "chromatic"},
  "260917_0179.wav": {"start": "C4", "pattern": "major"}
}
```

Run:
```bash
.venv/bin/python tools/make_fixtures.py /Users/j/SoundBanks/Ithaca/raw-samples-legacy-01 \
    /Users/j/SoundBanks/Ithaca/raw-samples-legacy-01/truth.json tests/fixtures/pitch
du -sh tests/fixtures/pitch
```
Expected: `260917_0180.wav: 24 úderů (3 kliků) → A0..G#2`, `260917_0179.wav: 29 úderů (…) → C4..C8`, celkem 53 souborů, ~9 MB. **Pokud počty nesedí (24 a 29), je to chyba detekce na reálných datech — oprav `detect.py` (typicky prahy nasazení/dělení) dřív, než pokračuješ; nesahej na truth.**

- [ ] **Step 5: Regresní test `tests/test_pitch_fixtures.py`**

```python
import json
from pathlib import Path
import pytest
from sample_slicer.io import read_wav
from sample_slicer.pitch import estimate_pitch

FIX = Path(__file__).parent / "fixtures" / "pitch"
TRUTH = json.loads((FIX / "truth.json").read_text())

@pytest.mark.parametrize("name", sorted(TRUTH))
def test_fixture_note(name):
    data, info = read_wav(FIX / name)
    p = estimate_pitch(data[:, 0], info.sample_rate)
    assert p is not None
    truth = TRUTH[name]
    diff = p.midi - truth
    # laťka detektoru: správná oktáva a do ±90 c (rozladěné B7/C8 řeší ladicí křivka)
    assert abs(diff) < 0.9, f"{name}: midi {p.midi:.2f} vs {truth} ({100*diff:+.0f} c), conf {p.confidence:.2f}"

def test_no_octave_errors_overall():
    bad = []
    for name, truth in TRUTH.items():
        data, info = read_wav(FIX / name)
        p = estimate_pitch(data[:, 0], info.sample_rate)
        if p is None or abs(p.midi - truth) >= 0.9:
            bad.append((name, None if p is None else round(p.midi, 2), truth))
    assert bad == []
```

Run: `.venv/bin/pytest tests/test_pitch_fixtures.py -q`
Expected: `54 passed`. Pokud některý fixture selže, srovnej se spikem (viz spec §4: A0–G#2 24/24, C4–C8 27/29 při ±50 c, B7/C8 +79/+73 c) a lad `pitch.py`, ne test.

- [ ] **Step 6: Commit**

```bash
git add sample_slicer/notes.py tools/make_fixtures.py tests/test_notes.py tests/test_pitch_fixtures.py tests/fixtures/pitch
git commit -m "test(pitch): fixtures z reálných nahrávek Petrof + regresní laťka; notes.py"
```

---

### Task 10: Ladicí křivka a přiřazení not (`tuning.py`)

**Files:**
- Create: `sample_slicer/tuning.py`, `tests/test_tuning.py`

**Interfaces:**
- Consumes: `pitch.Pitch`.
- Produces:
  - `@dataclass TuningParams(anchor_cents=35.0, accept_cents=50.0, min_confidence=0.5, curve_halfwidth=6, midi_lo=21, midi_hi=108)`.
  - `fit_tuning_curve(anchors: list[tuple[int, float]], halfwidth: int = 6) -> Callable[[float], float]` — `(midi, cents)` kotvy → funkce `midi → cents`; klouzavý medián kotev v ±halfwidth půltónech, lineární interpolace mezi kotvami, konstantní extrapolace; bez kotev vrací `lambda m: 0.0`.
  - `@dataclass Assignment` (viz Mapa souborů).
  - `assign_notes(pitches: list[Pitch | None], p: TuningParams = TuningParams()) -> tuple[list[Assignment], Callable]` — dva průchody dle specu §5; `None` pitch → `Assignment(None, 0, 0, False, "low_confidence")`.

- [ ] **Step 1: Failing testy**

```python
# tests/test_tuning.py
from sample_slicer.pitch import Pitch
from sample_slicer.tuning import fit_tuning_curve, assign_notes, TuningParams

def P(midi, conf=0.9):
    return Pitch(f0_hz=440.0 * 2 ** ((midi - 69) / 12), midi=midi, confidence=conf, n_votes=3, evidence=3)

def test_curve_constant_extrapolation_and_interpolation():
    curve = fit_tuning_curve([(60, 0.0), (72, 20.0)], halfwidth=0)
    assert curve(50) == 0.0 and curve(80) == 20.0 and abs(curve(66) - 10.0) < 1e-9

def test_curve_median_smoothing():
    curve = fit_tuning_curve([(60, 0.0), (61, 100.0), (62, 0.0)], halfwidth=1)
    assert curve(61) == 0.0        # osamělá odchylka se vyhladí

def test_assign_sharp_treble_via_curve():
    # střed přesný, od A7 rostoucí odchylka: 105 +46 c, 107 +79 c, 108 +73 c
    pitches = [P(60.0), P(72.05), P(84.1), P(96.2), P(105.46), P(107.79), P(108.73)]
    asg, curve = assign_notes(pitches)
    assert [a.midi for a in asg] == [60, 72, 84, 96, 105, 107, 108]
    assert asg[0].anchor and asg[4].anchor is False and asg[5].reason == "curve"
    assert abs(asg[5].cents_et - 79) < 1

def test_reject_out_of_tolerance_and_range():
    asg, _ = assign_notes([P(60.0), P(64.5), P(20.0), None, P(66.0, conf=0.2)])
    assert asg[1].midi is None and asg[1].reason == "out_of_tolerance"
    assert asg[2].midi is None and asg[2].reason == "out_of_range"
    assert asg[3].midi is None and asg[3].reason == "low_confidence"
    assert asg[4].midi is None and asg[4].reason == "low_confidence"
```

- [ ] **Step 2: Ověřit selhání** — `.venv/bin/pytest tests/test_tuning.py -q` → ImportError.

- [ ] **Step 3: Implementace `sample_slicer/tuning.py`**

```python
"""Ladicí křivka piana (odchylka v centech vs. MIDI) a přiřazení not ve dvou průchodech."""
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
    curve_halfwidth: int = 6
    midi_lo: int = 21
    midi_hi: int = 108


@dataclass
class Assignment:
    midi: int | None
    cents_et: float
    cents_curve: float
    anchor: bool
    reason: str


def fit_tuning_curve(anchors: list[tuple[int, float]], halfwidth: int = 6) -> Callable[[float], float]:
    if not anchors:
        return lambda m: 0.0
    by_midi: dict[int, list[float]] = {}
    for m, c in anchors:
        by_midi.setdefault(m, []).append(c)
    xs = np.array(sorted(by_midi))
    raw = np.array([np.median(by_midi[m]) for m in xs])
    smoothed = np.array([np.median(raw[(xs >= m - halfwidth) & (xs <= m + halfwidth)]) for m in xs])

    def curve(m: float) -> float:
        return float(np.interp(m, xs, smoothed))   # np.interp extrapoluje konstantně
    return curve


def assign_notes(pitches: list[Pitch | None], p: TuningParams = TuningParams()):
    anchors: list[tuple[int, float]] = []
    for pt in pitches:
        if pt is None or pt.confidence < p.min_confidence:
            continue
        m = int(round(pt.midi))
        if p.midi_lo <= m <= p.midi_hi and abs(pt.cents_et) <= p.anchor_cents:
            anchors.append((m, pt.cents_et))
    curve = fit_tuning_curve(anchors, p.curve_halfwidth)
    out: list[Assignment] = []
    for pt in pitches:
        if pt is None or pt.confidence < p.min_confidence:
            out.append(Assignment(None, 0.0, 0.0, False, "low_confidence"))
            continue
        m_et = int(round(pt.midi))
        if abs(pt.cents_et) <= p.anchor_cents and p.midi_lo <= m_et <= p.midi_hi:
            out.append(Assignment(m_et, pt.cents_et, pt.cents_et - curve(m_et), True, "anchor"))
            continue
        corrected = pt.midi - curve(pt.midi) / 100.0
        m = int(round(corrected))
        cents_curve = 100.0 * (corrected - m)
        cents_et = 100.0 * (pt.midi - m)
        if not (p.midi_lo <= m <= p.midi_hi):
            out.append(Assignment(None, cents_et, cents_curve, False, "out_of_range"))
        elif abs(cents_curve) > p.accept_cents:
            out.append(Assignment(None, cents_et, cents_curve, False, "out_of_tolerance"))
        else:
            out.append(Assignment(m, cents_et, cents_curve, False, "curve"))
    return out, curve
```

- [ ] **Step 4: Testy projdou** — `.venv/bin/pytest tests/test_tuning.py -q` → `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add sample_slicer/tuning.py tests/test_tuning.py
git commit -m "feat(tuning): ladicí křivka z kotev a přiřazení not ve dvou průchodech"
```

---

### Task 11: CLI `analyze` (dry-run, `--truth`, přesnost po oktávách)

**Files:**
- Create: `sample_slicer/analyze.py`, `tests/test_analyze.py`
- Modify: `sample_slicer/cli.py` (podpříkaz `analyze`)

**Interfaces:**
- Consumes: `slicing.slice_file`, `pitch.estimate_pitch`, `tuning.assign_notes`, `notes.expand_truth`, `notes.midi_to_name`.
- Produces:
  - `@dataclass HitRow(source: str, index: int, t_s: float, dur_s: float, peak_db: float, end_reason: str, pitch: Pitch | None, assignment: Assignment, truth: int | None)`.
  - `analyze_file(path, params, tail_s, truth_entry: dict | None) -> tuple[list[HitRow], list[Segment]]`.
  - `analyze_dir(src_dir, params, tail_s, truth: dict | None, log=print) -> list[HitRow]`.
  - `accuracy_by_octave(rows) -> dict[str, dict[str, int]]` — klíč `"A0-G#1"` styl `f"{lo}-{hi}"` po oktávách MIDI (21–32, 33–44, …), hodnoty `{"ok":, "octave":, "other":}`; klasifikace: `ok` = přiřazené MIDI == truth; `octave` = rozdíl násobek 12; `other`.
  - `format_rows(rows) -> str` (textová tabulka pro stdout).

- [ ] **Step 1: Failing test**

```python
# tests/test_analyze.py
import json, numpy as np
from sample_slicer.io import write_wav
from sample_slicer.analyze import analyze_dir, accuracy_by_octave
from sample_slicer.detect import DetectParams
from sample_slicer.cli import main
from tests.synth import piano_note, place, noise_floor

SR = 96000

def make_src(tmp_path, midis=(60, 62, 64)):
    src = tmp_path / "src"; src.mkdir()
    c = noise_floor(int((2 + 3 * len(midis)) * SR), SR, db=-85)
    for i, m in enumerate(midis):
        place(c, piano_note(m, SR, 2.0, decay_db_s=-20, inharmonicity=0.0003), 1.0 + 3.0 * i, SR)
    write_wav(src / "rec.wav", c[:, None], SR, 24)
    (src / "truth.json").write_text(json.dumps({"rec.wav": {"start": "C4", "pattern": "major"}}))
    return src

def test_analyze_with_truth(tmp_path):
    src = make_src(tmp_path)
    truth = json.loads((src / "truth.json").read_text())
    rows = analyze_dir(src, DetectParams(), 2.0, truth, log=lambda *_: None)
    assert [r.assignment.midi for r in rows] == [60, 62, 64]
    assert [r.truth for r in rows] == [60, 62, 64]
    acc = accuracy_by_octave(rows)
    assert sum(v["ok"] for v in acc.values()) == 3 and sum(v["octave"] + v["other"] for v in acc.values()) == 0

def test_cli_analyze_prints_and_writes_nothing(tmp_path, capsys):
    src = make_src(tmp_path)
    before = sorted(p.name for p in src.iterdir())
    assert main(["analyze", str(src), "--truth", str(src / "truth.json")]) == 0
    out = capsys.readouterr().out
    assert "C4" in out and "ok" in out
    assert sorted(p.name for p in src.iterdir()) == before
```

- [ ] **Step 2: Ověřit selhání** — `.venv/bin/pytest tests/test_analyze.py -q` → ImportError.

- [ ] **Step 3: Implementace `sample_slicer/analyze.py`**

```python
"""Dry-run analýza: segmenty + výška + přiřazení; s pravdou i přesnost po oktávách."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from .detect import DetectParams, Segment
from .io import UnsupportedWav, to_mono
from .notes import expand_truth, midi_to_name
from .pitch import Pitch, estimate_pitch
from .slicing import slice_file
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
                 tuning: TuningParams = TuningParams()):
    slices, rejected = slice_file(path, params, tail_s)
    pitches = [estimate_pitch(to_mono(s.audio)[s.segment.onset - s.segment.start:], s.sr) for s in slices]
    assignments, _ = assign_notes(pitches, tuning)
    truths = expand_truth(truth_entry, len(slices)) if truth_entry else [None] * len(slices)
    rows = [HitRow(Path(path).name, s.index, s.segment.onset / s.sr, len(s.audio) / s.sr, s.segment.peak_db,
                   s.segment.end_reason, pt, a, tr)
            for s, pt, a, tr in zip(slices, pitches, assignments, truths)]
    return rows, rejected


def analyze_dir(src_dir, params: DetectParams, tail_s: float, truth: dict | None, log=print):
    rows: list[HitRow] = []
    for wav in sorted(Path(src_dir).glob("*.[wW][aA][vV]")):
        entry = (truth or {}).get(wav.name)
        try:
            r, rejected = analyze_file(wav, params, tail_s, entry)
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
```

- [ ] **Step 4: Podpříkaz `analyze` v `cli.py`**

Do `build_parser()` přidej:

```python
    a = sub.add_parser("analyze", help="dry-run: segmenty, noty, confidence; s --truth přesnost")
    a.add_argument("src_dir")
    a.add_argument("--truth", default=None, help="JSON se známým pořadím nahrávání")
    add_detect_args(a)
    a.set_defaults(func=cmd_analyze)
```

a funkci:

```python
def cmd_analyze(args) -> int:
    import json
    from . import analyze as an
    truth = json.loads(open(args.truth).read()) if args.truth else None
    rows = an.analyze_dir(args.src_dir, detect_params_from_args(args), args.tail_s, truth)
    print(an.format_rows(rows))
    if truth:
        acc = an.accuracy_by_octave(rows)
        print("\npřesnost po oktávách (ok / oktávová chyba / jiná):")
        for k, v in acc.items():
            print(f"  {k:<8} ok {v['ok']:3d}  octave {v['octave']:3d}  other {v['other']:3d}")
    return 0
```

- [ ] **Step 5: Testy projdou** — `.venv/bin/pytest -q` → vše zelené.

- [ ] **Step 6: Ověření na reálných datech (ruční)**

Run:
```bash
.venv/bin/sample-slicer analyze /Users/j/SoundBanks/Ithaca/raw-samples-legacy-01 \
    --truth /Users/j/SoundBanks/Ithaca/raw-samples-legacy-01/truth.json | tail -12
```
Expected: `A0-G#1 ok 12`, `A1-G#2 ok 12`, C4–C8 všechny `ok` (29), 0 octave, 0 other. Když ne, oprav pitch/tuning (ne test) a vrať se sem.

- [ ] **Step 7: Commit**

```bash
git add sample_slicer/analyze.py sample_slicer/cli.py tests/test_analyze.py
git commit -m "feat(analyze): dry-run s přiřazením not a přesností proti pravdě"
```

---

### Task 12: Stavba banky (`bank.py`): layout, index, ffmpeg, rejected, overrides, report

**Files:**
- Create: `sample_slicer/bank.py`, `tests/test_bank.py`

**Interfaces:**
- Consumes: `analyze.analyze_file` (HitRow), `slicing.slice_file` (Slice audio), `io.write_wav`, `notes.midi_to_name`.
- Produces:
  - `md5_16(path) -> str`.
  - `convert_48k16(src_wav, dst_wav, retune_cents: float = 0.0) -> None` — ffmpeg; `retune_cents ≠ 0` přidá `asetrate=<sr*2^(-cents/1200)>` před `aresample` (posun výšky změnou poměru).
  - `ffmpeg_available() -> bool`.
  - `class BankIndex` — soubor `<original>/.slicer-index.json`: `{"version": 1, "sources": {"<name>": {"md5": "...", "params": "<hash>", "entries": {"<idx>": {"orig": "m060/abcd....wav", "out": "m060/....wav", "midi": 60, "cents_et": 3.0, "cents_curve": 1.0, "confidence": 0.9, "t_s": 6.85, "reason": "anchor"}}}}}`; metody `load(path)`, `save()`, `source_up_to_date(name, md5, params_hash) -> bool`, `remove_source_files(name, original_dir, out_dir)`, `set_source(name, md5, params_hash, entries)`.
  - `params_hash(detect_params, tail_s, tuning_params, retune: bool) -> str` (sha1 z repr).
  - `@dataclass BuildResult(written: int, rejected: int, skipped_sources: int, rows: list[HitRow])`.
  - `build_bank(src_dir, original_dir, out_dir, params, tail_s, tuning, retune=False, overrides: dict | None = None, log=print) -> BuildResult`.
  - `write_report(original_dir, rows, index) -> Path` — `<original>/report.md`.

- [ ] **Step 1: Failing testy**

```python
# tests/test_bank.py
import json, shutil, numpy as np, pytest
from pathlib import Path
from sample_slicer.io import write_wav, read_wav
from sample_slicer.bank import build_bank, md5_16, ffmpeg_available, BankIndex
from sample_slicer.detect import DetectParams
from sample_slicer.tuning import TuningParams
from tests.synth import piano_note, place, noise_floor

SR = 96000
needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg chybí")

def make_src(tmp_path, midis=(60, 62), name="rec.wav"):
    src = tmp_path / "src"; src.mkdir(exist_ok=True)
    c = noise_floor(int((2 + 3 * len(midis)) * SR), SR, db=-85)
    for i, m in enumerate(midis):
        place(c, piano_note(m, SR, 2.0, decay_db_s=-20, inharmonicity=0.0003), 1.0 + 3.0 * i, SR)
    write_wav(src / name, np.stack([c, c], axis=1), SR, 24)
    return src

def run(src, tmp_path, **kw):
    return build_bank(src, tmp_path / "orig", tmp_path / "out", DetectParams(), 2.0, TuningParams(),
                      log=lambda *_: None, **kw)

@needs_ffmpeg
def test_build_layout_and_formats(tmp_path):
    src = make_src(tmp_path)
    res = run(src, tmp_path)
    assert res.written == 2 and res.rejected == 0
    orig = sorted((tmp_path / "orig").glob("m*/*.wav")); out = sorted((tmp_path / "out").glob("m*/*.wav"))
    assert [p.parent.name for p in orig] == ["m060", "m062"] and len(out) == 2
    assert all(len(p.stem) == 16 for p in orig + out)
    assert orig[0].stem == md5_16(orig[0])
    d, info = read_wav(orig[0]); assert info.sample_rate == SR and info.bits == 24 and info.channels == 2
    d, info = read_wav(out[0]); assert info.sample_rate == 48000 and info.bits == 16 and info.channels == 2
    assert (tmp_path / "orig" / "report.md").exists() and (tmp_path / "orig" / ".slicer-index.json").exists()

@needs_ffmpeg
def test_build_is_idempotent_and_keeps_foreign_files(tmp_path):
    src = make_src(tmp_path)
    run(src, tmp_path)
    foreign = tmp_path / "orig" / "m060" / "manual.wav"; foreign.write_bytes(b"x")
    snapshot = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.wav"))
    res = run(src, tmp_path)
    assert res.skipped_sources == 1 and res.written == 0
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.wav")) == snapshot
    assert foreign.exists()

@needs_ffmpeg
def test_changed_params_replace_old_files(tmp_path):
    src = make_src(tmp_path)
    run(src, tmp_path)
    old = sorted((tmp_path / "orig").glob("m*/*.wav"))
    res = build_bank(src, tmp_path / "orig", tmp_path / "out", DetectParams(), 1.0, TuningParams(), log=lambda *_: None)
    new = sorted((tmp_path / "orig").glob("m*/*.wav"))
    assert res.written == 2 and len(new) == 2 and set(new).isdisjoint(set(old))

@needs_ffmpeg
def test_overrides_skip_and_midi(tmp_path):
    src = make_src(tmp_path, midis=(60, 62, 64))
    res = run(src, tmp_path, overrides={"rec.wav": {"skip": [1], "midi": {"2": 70}}})
    assert res.written == 2
    assert sorted(p.name for p in (tmp_path / "orig").glob("m*")) == ["m060", "m070"]
    assert "override" in (tmp_path / "orig" / "report.md").read_text()

@needs_ffmpeg
def test_rejected_goes_to_folder(tmp_path):
    src = make_src(tmp_path, midis=(60,))
    # druhý „úder" = čistý šum s hlasitostí tónu → nízká confidence → _rejected
    data, info = read_wav(src / "rec.wav")
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal((SR, 2)) * 0.1).astype(np.float32)
    write_wav(src / "rec.wav", np.concatenate([data, noise, np.zeros((SR, 2), np.float32)]), SR, 24)
    res = run(src, tmp_path)
    assert res.rejected >= 1
    assert list((tmp_path / "orig" / "_rejected").glob("rec_*.wav"))
```

- [ ] **Step 2: Ověřit selhání** — `.venv/bin/pytest tests/test_bank.py -q` → ImportError.

- [ ] **Step 3: Implementace `sample_slicer/bank.py`**

```python
"""Stavba banky: layout m###/<hash>.wav, index pro idempotenci, ffmpeg převod, rejected, overrides, report."""
from __future__ import annotations
import hashlib, json, shutil, subprocess, tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from .analyze import HitRow, analyze_file
from .detect import DetectParams
from .io import UnsupportedWav, write_wav
from .notes import midi_to_name
from .tuning import Assignment, TuningParams

INDEX_NAME = ".slicer-index.json"


def md5_16(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def convert_48k16(src_wav, dst_wav, retune_cents: float = 0.0, src_sr: int | None = None) -> None:
    af = "aresample=48000:resampler=soxr:precision=28:dither_method=triangular"
    if retune_cents and src_sr:
        af = f"asetrate={src_sr * 2 ** (-retune_cents / 1200):.3f}," + af
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src_wav), "-af", af, "-c:a", "pcm_s16le", str(dst_wav)]
    subprocess.run(cmd, check=True)


def params_hash(params: DetectParams, tail_s: float, tuning: TuningParams, retune: bool) -> str:
    return hashlib.sha1(repr((asdict(params), tail_s, asdict(tuning), retune)).encode()).hexdigest()[:12]


class BankIndex:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"version": 1, "sources": {}}

    @classmethod
    def load(cls, original_dir) -> "BankIndex":
        ix = cls(Path(original_dir) / INDEX_NAME)
        if ix.path.exists():
            ix.data = json.loads(ix.path.read_text())
        return ix

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))

    def source_up_to_date(self, name: str, md5: str, phash: str) -> bool:
        s = self.data["sources"].get(name)
        return bool(s) and s["md5"] == md5 and s["params"] == phash

    def remove_source_files(self, name: str, original_dir, out_dir) -> None:
        s = self.data["sources"].pop(name, None)
        if not s:
            return
        for e in s["entries"].values():
            for base, key in ((original_dir, "orig"), (out_dir, "out")):
                p = Path(base) / e[key]
                if p.exists():
                    p.unlink()
                if p.parent.exists() and not any(p.parent.iterdir()):
                    p.parent.rmdir()

    def set_source(self, name: str, md5: str, phash: str, entries: dict) -> None:
        self.data["sources"][name] = {"md5": md5, "params": phash, "entries": entries}


@dataclass
class BuildResult:
    written: int
    rejected: int
    skipped_sources: int
    rows: list


def _store_hashed(audio, sr, bits, target_dir: Path) -> str:
    """Zapíše audio do target_dir/<md5-16>.wav (přes temp soubor); vrátí název souboru."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=target_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    write_wav(tmp_path, audio, sr, bits)
    name = md5_16(tmp_path) + ".wav"
    tmp_path.replace(target_dir / name)
    return name


def build_bank(src_dir, original_dir, out_dir, params: DetectParams, tail_s: float,
               tuning: TuningParams, retune: bool = False, overrides: dict | None = None, log=print) -> BuildResult:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg nenalezen v PATH — je potřeba pro převod na 48 kHz / 16 bit")
    src_dir, original_dir, out_dir = Path(src_dir), Path(original_dir), Path(out_dir)
    overrides = overrides or {}
    index = BankIndex.load(original_dir)
    phash = params_hash(params, tail_s, tuning, retune)
    written = rejected = skipped = 0
    all_rows: list[HitRow] = []
    from .slicing import slice_file
    for wav in sorted(src_dir.glob("*.[wW][aA][vV]")):
        md5 = md5_16(wav)
        if index.source_up_to_date(wav.name, md5, phash):
            log(f"{wav.name}: beze změny, přeskočeno")
            skipped += 1
            continue
        index.remove_source_files(wav.name, original_dir, out_dir)
        try:
            rows, _ = analyze_file(wav, params, tail_s, None, tuning)
            slices, _ = slice_file(wav, params, tail_s)
        except UnsupportedWav as e:
            log(f"PŘESKOČENO {wav.name}: {e}")
            continue
        ov = overrides.get(wav.name, {})
        entries = {}
        for row, s in zip(rows, slices):
            if row.index in ov.get("skip", []):
                row.assignment = Assignment(None, row.assignment.cents_et, row.assignment.cents_curve, False, "override")
            elif str(row.index) in ov.get("midi", {}):
                m = int(ov["midi"][str(row.index)])
                row.assignment = Assignment(m, row.assignment.cents_et, row.assignment.cents_curve, False, "override")
            a = row.assignment
            if a.midi is None:
                rejected += 1
                rej_dir = original_dir / "_rejected"; rej_dir.mkdir(parents=True, exist_ok=True)
                write_wav(rej_dir / f"{wav.stem}_{row.index:02d}_{a.reason}.wav", s.audio, s.sr, s.bits)
                continue
            mdir = f"m{a.midi:03d}"
            orig_name = _store_hashed(s.audio, s.sr, s.bits, original_dir / mdir)
            (out_dir / mdir).mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=out_dir / mdir, delete=False) as tmp:
                tmp_out = Path(tmp.name)
            convert_48k16(original_dir / mdir / orig_name, tmp_out, a.cents_et if retune else 0.0, s.sr)
            out_name = md5_16(tmp_out) + ".wav"
            tmp_out.replace(out_dir / mdir / out_name)
            entries[str(row.index)] = {"orig": f"{mdir}/{orig_name}", "out": f"{mdir}/{out_name}", "midi": a.midi,
                                       "cents_et": round(a.cents_et, 1), "cents_curve": round(a.cents_curve, 1),
                                       "confidence": round(row.pitch.confidence, 3) if row.pitch else 0.0,
                                       "t_s": round(row.t_s, 3), "reason": a.reason}
            written += 1
        index.set_source(wav.name, md5, phash, entries)
        index.save()
        all_rows.extend(rows)
        log(f"{wav.name}: {len(entries)} zapsáno, {len(rows) - len(entries)} odmítnuto")
    write_report(original_dir, all_rows, index)
    return BuildResult(written, rejected, skipped, all_rows)


def write_report(original_dir, rows: list[HitRow], index: BankIndex) -> Path:
    from .analyze import format_rows
    lines = ["# Report stavby banky", "", "## Údery (tento běh)", "", "```", format_rows(rows), "```", ""]
    rej = [r for r in rows if r.assignment.midi is None]
    lines += ["## Odmítnuté", ""] + ([f"- {r.source} #{r.index} @ {r.t_s:.2f}s: {r.assignment.reason}" for r in rej] or ["- žádné"]) + [""]
    layers: dict[int, int] = {}
    for s in index.data["sources"].values():
        for e in s["entries"].values():
            layers[e["midi"]] = layers.get(e["midi"], 0) + 1
    lines += ["## Vrstvy na notu (celá banka dle indexu)", "", "| nota | midi | vrstev |", "|---|---|---|"]
    lines += [f"| {midi_to_name(m)} | {m} | {layers[m]} |" for m in sorted(layers)]
    holes = [m for m in range(21, 109) if m not in layers]
    lines += ["", f"Díry na klaviatuře ({len(holes)}): " + (", ".join(midi_to_name(m) for m in holes) if holes else "žádné"), ""]
    tuning_rows = sorted((e["midi"], e["cents_et"]) for s in index.data["sources"].values() for e in s["entries"].values())
    lines += ["## Ladění (centy vs. temperované)", "", "| nota | centy |", "|---|---|"] + [f"| {midi_to_name(m)} | {c:+.0f} |" for m, c in tuning_rows] + [""]
    path = Path(original_dir) / "report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path
```

Pozn.: `analyze_file` a `slice_file` se v `build_bank` volají dvakrát na tentýž soubor (analýza + audio). To čte 600 MB soubor dvakrát; pokud to při ručním testu na reálných datech trvá > 2 min na soubor, uprav `analyze_file`, aby vracela i `slices` (`return rows, rejected, slices`) a `build_bank` použij ty.

- [ ] **Step 4: Testy projdou** — `.venv/bin/pytest tests/test_bank.py -q` → `5 passed` (nebo `5 skipped` bez ffmpeg — na tomto stroji ffmpeg je, takže musí být passed).

- [ ] **Step 5: Commit**

```bash
git add sample_slicer/bank.py tests/test_bank.py
git commit -m "feat(bank): layout m###/<hash>, index, ffmpeg 48k/16b, rejected, overrides, report"
```

---

### Task 13: CLI `build`, README, běh na reálných datech

**Files:**
- Modify: `sample_slicer/cli.py` (podpříkaz `build`), `README.md`, `tests/test_slicing.py` (přidat CLI test pro build)

**Interfaces:**
- Consumes: `bank.build_bank`.
- Produces: `sample-slicer build <src-dir> --original <dir> --out <dir> [--retune] [--overrides overrides.json] [detekční přepínače]`. Když `--overrides` není zadán a v `<src-dir>/overrides.json` existuje, použije se automaticky.

- [ ] **Step 1: Failing CLI test (přidat do `tests/test_slicing.py`)**

```python
import pytest
from sample_slicer.bank import ffmpeg_available

@pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg chybí")
def test_cli_build(tmp_path):
    src = tmp_path / "src"; src.mkdir(); make_file(src / "rec.wav")
    assert main(["build", str(src), "--original", str(tmp_path / "o"), "--out", str(tmp_path / "b")]) == 0
    assert len(list((tmp_path / "o").glob("m*/*.wav"))) == 2
    assert len(list((tmp_path / "b").glob("m*/*.wav"))) == 2
```

- [ ] **Step 2: Podpříkaz `build` v `cli.py`**

```python
def cmd_build(args) -> int:
    import json
    from pathlib import Path
    from .bank import build_bank
    from .tuning import TuningParams
    ov_path = Path(args.overrides) if args.overrides else Path(args.src_dir) / "overrides.json"
    overrides = json.loads(ov_path.read_text()) if ov_path.exists() else None
    try:
        res = build_bank(args.src_dir, args.original, args.out, detect_params_from_args(args), args.tail_s,
                         TuningParams(), retune=args.retune, overrides=overrides)
    except RuntimeError as e:
        print(f"CHYBA: {e}", file=sys.stderr)
        return 2
    print(f"Zapsáno {res.written} úderů, odmítnuto {res.rejected}, přeskočeno zdrojů {res.skipped_sources}. "
          f"Report: {Path(args.original) / 'report.md'}")
    return 0
```

a do `build_parser()`:

```python
    b = sub.add_parser("build", help="celý workflow: zdroje → original (96k/24b) → out (48k/16b)")
    b.add_argument("src_dir")
    b.add_argument("--original", required=True); b.add_argument("--out", required=True)
    b.add_argument("--retune", action="store_true", help="dolaď výšku na temperované ladění při převodu")
    b.add_argument("--overrides", default=None, help="JSON s ručními zásahy (výchozí <src>/overrides.json)")
    add_detect_args(b)
    b.set_defaults(func=cmd_build)
```

- [ ] **Step 3: Testy projdou** — `.venv/bin/pytest -q` → vše zelené.

- [ ] **Step 4: README**

Doplň do `README.md` sekci „Stavba banky pro ithaca-legacy":

```markdown
## Stavba banky pro ithaca-legacy

    sample-slicer analyze <raw-dir> [--truth truth.json]      # dry-run, nic nezapisuje
    sample-slicer build <raw-dir> --original <orig> --out <bank>

- `<orig>/m###/<hash>.wav` – ořezané údery v původním formátu (např. 96 kHz / 24 bit)
- `<bank>/m###/<hash>.wav` – 48 kHz / 16 bit (ffmpeg soxr + dither), tohle načítá ithaca
- `<orig>/report.md` – tabulka úderů, odmítnuté, vrstvy na notu, ladění
- `<orig>/_rejected/` – údery bez spolehlivé noty
- `<orig>/.slicer-index.json` – idempotence: opakovaný běh nic nezdvojí, nové nahrávky se přidají
- `<raw-dir>/overrides.json` – ruční zásahy: `{"rec.wav": {"skip": [17], "midi": {"3": 24}}}`

Nota se určuje jen z audia (viz spec `docs/superpowers/specs/2026-09-21-bank-pipeline-design.md`).
`--truth` slouží jen k měření přesnosti proti známému pořadí nahrávání.
Vyžaduje `ffmpeg` v PATH.
```

- [ ] **Step 5: Běh na reálných datech (ruční ověření)**

Run:
```bash
.venv/bin/sample-slicer build /Users/j/SoundBanks/Ithaca/raw-samples-legacy-01 \
    --original /Users/j/SoundBanks/Ithaca/ap-petrof-dynamic-original \
    --out /Users/j/SoundBanks/Ithaca/ap-petrof-dynamic
ls /Users/j/SoundBanks/Ithaca/ap-petrof-dynamic | head; ls /Users/j/SoundBanks/Ithaca/ap-petrof-dynamic | wc -l
sed -n 1,60p /Users/j/SoundBanks/Ithaca/ap-petrof-dynamic-original/report.md
```
Expected: 53 zapsaných úderů, 0 odmítnutých (nebo jen B7/C8 pokud křivka nestačí — pak je to nález do reportu, ne bug plánu), složky `m021`…`m044` a `m060`…`m108` (C dur), každá s jedním souborem 48 kHz / 16 bit. Druhý běh téhož příkazu: `přeskočeno zdrojů 2`, žádné nové soubory.

Pak v ithaca-legacy (`~/Projects/ithaca-legacy`): spustit `ithaca-gui`, načíst banku `/Users/j/SoundBanks/Ithaca/ap-petrof-dynamic` a zahrát A0–G#2 a C4–C8. Očekávání: každá nota zní správnou výškou, nasazení bez cvaknutí, dozvuk končí plynule. Poznámky zapsat do reportu ručně jen jako komentář v chatu (do banky se nic ručně nepřidává).

- [ ] **Step 6: Commit**

```bash
git add sample_slicer/cli.py README.md tests/test_slicing.py
git commit -m "feat(cli): build — celý workflow; README"
```

---

## Self-review (provedeno při psaní)

- **Spec §1–§2** (cíl, umístění, struktura): Task 1, 7, 13. `slicing.py` místo `slice.py` (aby nekolidovalo s builtin `slice`); spec to nevylučuje.
- **§3 detekce**: Task 3 (obálka, dno, sklon), Task 5 (nasazení, dělení, kliky, konec s 4 důvody), Task 6 (dozvuk, fade-in). Pravidlo `sustained` u dělení (nová část drží ≥ 0,5 s) je v `find_onsets`.
- **§4 pitch**: Task 8 (všechny kroky 1–5 včetně confidence), Task 9 (reálná laťka).
- **§5 přiřazení**: Task 10; kolize téže noty se nezakazují (bank.py je ukládá jako další soubory v `m###/`).
- **§6 banka**: Task 12 (layout, hash, ffmpeg, retune, index, report, rejected, overrides, cizí soubory se nemažou — `remove_source_files` maže jen cesty z indexu).
- **§7 CLI**: Task 7 (`slice`), 11 (`analyze --truth`), 13 (`build`); `truth.json` formát v Task 9 (`notes.expand_truth`).
- **§8 chyby**: `UnsupportedWav` per soubor (Task 2, 7, 11, 12), ffmpeg check (Task 12/13), nikdy tiché přiřazení (Task 10 vrací `None` + důvod).
- **§9 testy**: pitch fixtures (Task 9), syntetická detekce (Task 5), tail (Task 6), I/O 24 bit bit-přesně (Task 2), idempotence + cizí soubor (Task 12), tuning +79 c (Task 10).
- **§10 etapy**: pořadí tasků odpovídá.
- **Typy**: `Segment` (Task 5) používají Task 6, 7, 11; `Pitch` (Task 8) používají Task 10, 11; `Assignment` (Task 10) používají Task 11, 12; `HitRow` (Task 11) používá Task 12; `DetectParams` všude přes `cli.add_detect_args`.
- **Známé místo k úpravě při exekuci**: v Tasku 12 dvojí čtení zdroje (poznámka pod kódem) — optimalizovat jen když je to na reálných datech pomalé.

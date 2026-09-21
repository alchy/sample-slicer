"""Stavba banky: layout m###/<hash>.wav, index pro idempotenci, ffmpeg převod, rejected, overrides, report."""
from __future__ import annotations
import hashlib, json, shutil, subprocess, tempfile
from dataclasses import dataclass, asdict
from pathlib import Path
from .analyze import HitRow, analyze_file, format_rows
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


_ENGINE: str | None = None
AF_SOXR = "aresample=48000:resampler=soxr:precision=28:dither_method=triangular"
AF_SWR = "aresample=48000:resampler=swr:filter_size=256:cutoff=0.98:dither_method=triangular"


def resampler_engine(log=None) -> str:
    """'soxr' pokud ho ffmpeg umí (zkusí se jednou na krátkém tichu), jinak 'swr' (vestavěný,
    s filter_size=256 kvalitativně blízko; Homebrew ffmpeg bývá bez libsoxr)."""
    global _ENGINE
    if _ENGINE is None:
        with tempfile.TemporaryDirectory() as d:
            probe = Path(d) / "p.wav"
            import numpy as np
            write_wav(probe, np.zeros((960, 1), dtype=np.float32), 96000, 16)
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(probe), "-af", AF_SOXR,
                                "-c:a", "pcm_s16le", str(Path(d) / "o.wav")], capture_output=True)
        _ENGINE = "soxr" if r.returncode == 0 else "swr"
        if _ENGINE == "swr" and log:
            log("UPOZORNĚNÍ: ffmpeg bez libsoxr — resampling přes swresample (filter_size=256)")
    return _ENGINE


def convert_48k16(src_wav, dst_wav, retune_cents: float = 0.0, src_sr: int | None = None) -> None:
    """→ 48 kHz / 16 bit (soxr nebo swr) + triangulární dither; retune posune výšku změnou poměru."""
    af = AF_SOXR if resampler_engine() == "soxr" else AF_SWR
    if retune_cents and src_sr:
        af = f"asetrate={src_sr * 2 ** (-retune_cents / 1200):.3f}," + af
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src_wav), "-af", af, "-c:a", "pcm_s16le", str(dst_wav)]
    subprocess.run(cmd, check=True)


def params_hash(params: DetectParams, tail_s: float, tuning: TuningParams, retune: bool) -> str:
    return hashlib.sha1(repr((asdict(params), tail_s, asdict(tuning), retune)).encode()).hexdigest()[:12]


class BankIndex:
    """<original>/.slicer-index.json: zdroj → (md5, hash parametrů, údery → soubory). Maže jen to, co sám zapsal."""

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
    engine = resampler_engine(log)
    src_dir, original_dir, out_dir = Path(src_dir), Path(original_dir), Path(out_dir)
    overrides = overrides or {}
    index = BankIndex.load(original_dir)
    phash = params_hash(params, tail_s, tuning, retune)
    written = rejected = skipped = 0
    all_rows: list[HitRow] = []
    for wav in sorted(src_dir.glob("*.[wW][aA][vV]")):
        md5 = md5_16(wav)
        if index.source_up_to_date(wav.name, md5, phash):
            log(f"{wav.name}: beze změny, přeskočeno")
            skipped += 1
            continue
        index.remove_source_files(wav.name, original_dir, out_dir)
        try:
            rows, _, slices = analyze_file(wav, params, tail_s, None, tuning)
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
                rej_dir = original_dir / "_rejected"
                rej_dir.mkdir(parents=True, exist_ok=True)
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
        index.data["resampler"] = engine
        index.save()
        all_rows.extend(rows)
        log(f"{wav.name}: {len(entries)} zapsáno, {len(rows) - len(entries)} odmítnuto")
    write_report(original_dir, all_rows, index)
    return BuildResult(written, rejected, skipped, all_rows)


def write_report(original_dir, rows: list[HitRow], index: BankIndex) -> Path:
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

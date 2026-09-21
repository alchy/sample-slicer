"""Běh úlohy (analyze / build / slice) mimo GUI vlákno; log a výsledek přes signály."""
from __future__ import annotations
from pathlib import Path
from PySide6.QtCore import QThread, Signal
from sample_slicer.detect import DetectParams


def detect_params(p: dict) -> DetectParams:
    return DetectParams(end_level_db=float(p["end_level_db"]), max_len_s=float(p["max_len_s"]),
                        preroll_ms=float(p["preroll_ms"]))


class JobWorker(QThread):
    log = Signal(str, str)          # (zpráva, úroveň INFO/WARN/ERROR)
    done = Signal(str)              # shrnutí
    failed = Signal(str)

    def __init__(self, job: str, p: dict, parent=None):
        super().__init__(parent)
        self.job, self.p = job, p

    def _log(self, msg, level="INFO"):
        self.log.emit(str(msg), level)

    def run(self):
        try:
            self.done.emit(getattr(self, f"_{self.job}")())
        except Exception as e:                       # chyba úlohy nesmí shodit GUI
            self.failed.emit(f"{type(e).__name__}: {e}")

    def _analyze(self) -> str:
        from sample_slicer import analyze as an
        rows = an.analyze_dir(self.p["src"], detect_params(self.p), float(self.p["tail_s"]), None, log=self._log)
        self._log(an.format_rows(rows), "TABLE")
        rejected = [r for r in rows if r.assignment.midi is None]
        return f"Analýza: {len(rows)} úderů, z toho {len(rejected)} bez spolehlivé noty. Nic nebylo zapsáno."

    def _build(self) -> str:
        import json
        from sample_slicer.bank import build_bank
        from sample_slicer.tuning import TuningParams
        ov = Path(self.p["src"]) / "overrides.json"
        overrides = json.loads(ov.read_text()) if ov.exists() else None
        res = build_bank(self.p["src"], self.p["original"], self.p["bank"], detect_params(self.p),
                         float(self.p["tail_s"]), TuningParams(), retune=bool(self.p["retune"]),
                         overrides=overrides, log=self._log)
        return (f"Zapsáno {res.written} úderů, odmítnuto {res.rejected}, přeskočeno zdrojů {res.skipped_sources}. "
                f"Report: {Path(self.p['original']) / 'report.md'}")

    def _slice(self) -> str:
        from sample_slicer.slicing import slice_dir
        n = slice_dir(self.p["src"], self.p["out"], detect_params(self.p), float(self.p["tail_s"]),
                      float(self.p["fade_in_ms"]), log=self._log)
        return f"Uloženo {n} úderů do {self.p['out']}"

"""
Audio Processing Logic
======================
Core funkce pro detekci a zpracování audio segmentů.
Tenká vrstva nad balíčkem sample_slicer (detekce, dozvuk, I/O žijí tam).
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

logger = logging.getLogger(__name__)


# --- Statistiky zpracování ---
class ProcessingStats:
    """Statistiky pro sledování zpracování"""

    def __init__(self):
        self.files_processed = 0
        self.files_failed = 0
        self.segments_created = 0
        self.segments_skipped = 0
        self.total_input_duration = 0.0
        self.total_output_duration = 0.0
        self.audio_formats = {}

    def add_format(self, sample_rate: int, channels: int, bit_depth: int):
        """Zaznamenání audio formátu"""
        format_key = f"{sample_rate}Hz_{channels}ch_{bit_depth}bit"
        self.audio_formats[format_key] = self.audio_formats.get(format_key, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Export statistik jako dict"""
        return {
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "segments_created": self.segments_created,
            "segments_skipped": self.segments_skipped,
            "total_input_duration": self.total_input_duration,
            "total_output_duration": self.total_output_duration,
            "audio_formats": self.audio_formats
        }


# --- Validace ---
def validate_parameters(threshold_db: float, min_length: float, min_length_after_trim: float) -> List[str]:
    """
    Validuje zpracovací parametry.

    Returns:
        List chybových hlášek (prázdný = OK)
    """
    errors = []

    if threshold_db > 0:
        errors.append(f"Threshold must be negative or zero, got: {threshold_db}")

    if min_length < 0:
        errors.append(f"Min length must be positive, got: {min_length}")

    if min_length_after_trim < 0:
        errors.append(f"Min length after trim must be positive, got: {min_length_after_trim}")

    return errors


# --- Zpracování souboru přes sample_slicer ---
from sample_slicer.slicing import slice_file
from sample_slicer.io import write_wav, UnsupportedWav
from sample_slicer.detect import DetectParams


def process_wav_file(
    input_path: str,
    output_dir: str,
    threshold_db: float = -45.0,
    min_length: float = 3.0,
    min_length_after_trim: float = 0.5,
    trim_threshold_offset: float = 10.0,
    apply_fades: bool = True,
    fade_in_ms: float = 5.0,
    fade_out_percent: float = 20.0,
    overwrite: bool = True,
    stats: Optional[ProcessingStats] = None,
    progress_callback=None,
    log_callback=None
) -> bool:
    """
    Zpracuje jeden WAV soubor přes sample_slicer.slice_file.

    Mapování starých parametrů GUI: threshold_db → end_level_db (kde končí dozvuk),
    fade_in_ms → fade-in (jen když apply_fades). Ostatní staré prahy
    (min_length, min_length_after_trim, trim_threshold_offset, fade_out_percent)
    nový algoritmus nepotřebuje a ignorují se.

    Returns:
        True pokud úspěch
    """
    def log(msg: str, level: str = "INFO"):
        if log_callback:
            log_callback(msg, level)
        else:
            getattr(logger, level.lower())(msg)

    src = Path(input_path)
    log(f"Loading file: {src.name}")
    params = DetectParams(end_level_db=float(threshold_db))
    try:
        slices, rejected = slice_file(src, params, tail_s=2.0,
                                      fade_in_ms=float(fade_in_ms) if apply_fades else 0.0)
    except UnsupportedWav as e:
        log(str(e), "ERROR")
        if stats:
            stats.files_failed += 1
        return False

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(slices):
        start_ms = int(s.segment.start / s.sr * 1000)
        dur_ms = int(len(s.audio) / s.sr * 1000)
        out = Path(output_dir) / f"{src.stem}_slice_{s.index + 1:03d}_start_{start_ms}ms_dur_{dur_ms}ms.wav"
        if out.exists() and not overwrite:
            if stats:
                stats.segments_skipped += 1
            continue
        write_wav(out, s.audio, s.sr, s.bits)
        if stats:
            stats.segments_created += 1
            stats.total_output_duration += len(s.audio) / s.sr
        if progress_callback:
            progress_callback(100.0 * (i + 1) / max(1, len(slices)))
    log(f"{src.name}: {len(slices)} úderů, {len(rejected)} kliků zahozeno")
    if stats:
        if slices:
            stats.add_format(slices[0].sr, slices[0].channels, slices[0].bits)
        stats.files_processed += 1
    return True

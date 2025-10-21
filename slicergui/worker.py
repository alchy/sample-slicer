"""
Processing Worker
=================
QThread worker pro asynchronní zpracování audio souborů.
Zajišťuje, že GUI nebude během zpracování zamrzlé.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any

from PySide6.QtCore import QThread, Signal

from slicergui.logic import process_wav_file, ProcessingStats

logger = logging.getLogger(__name__)


class ProcessingWorker(QThread):
    """
    Worker thread pro zpracování WAV souborů.
    Emituje signály pro GUI update.
    """

    # Signály pro komunikaci s GUI
    progress_total = Signal(int)           # Celkový progress (0-100)
    progress_file = Signal(int)            # Progress aktuálního souboru (0-100)
    current_file = Signal(str)             # Název aktuálního souboru
    log_message = Signal(str, str)         # (message, level)
    finished = Signal(dict)                # Statistiky při dokončení
    error = Signal(str)                    # Chybová hláška

    def __init__(
        self,
        wav_files: List[Path],
        output_dir: str,
        params: Dict[str, Any]
    ):
        """
        Args:
            wav_files: Seznam WAV souborů k zpracování
            output_dir: Výstupní adresář
            params: Dict s processing parametry
        """
        super().__init__()

        self.wav_files = wav_files
        self.output_dir = output_dir
        self.params = params

        self._is_cancelled = False

    def run(self):
        """
        Hlavní metoda threadu - zpracovává soubory.
        Běží v samostatném threadu, proto nesmí přímo manipulovat GUI.
        """
        try:
            self._process_files()
        except Exception as e:
            logger.exception("Unexpected error in worker thread")
            self.error.emit(f"Unexpected error: {str(e)}")

    def _process_files(self):
        """Zpracování všech souborů"""
        stats = ProcessingStats()
        total_files = len(self.wav_files)

        self.log_message.emit(f"Starting processing of {total_files} files...", "INFO")

        for idx, wav_file in enumerate(self.wav_files):
            # Kontrola cancel flag
            if self._is_cancelled:
                self.log_message.emit("Processing cancelled by user", "WARNING")
                break

            # Update celkového progressu
            total_progress = int((idx / total_files) * 100)
            self.progress_total.emit(total_progress)

            # Oznámení aktuálního souboru
            self.current_file.emit(wav_file.name)
            self.log_message.emit(f"[{idx + 1}/{total_files}] Processing: {wav_file.name}", "INFO")

            # Reset file progress
            self.progress_file.emit(0)

            # Zpracování souboru
            success = process_wav_file(
                input_path=str(wav_file),
                output_dir=self.output_dir,
                threshold_db=self.params.get("threshold_db", -45.0),
                min_length=self.params.get("min_length", 3.0),
                min_length_after_trim=self.params.get("min_length_after_trim", 0.5),
                trim_threshold_offset=self.params.get("trim_threshold_offset", 10.0),
                apply_fades=self.params.get("apply_fades", True),
                fade_in_ms=self.params.get("fade_in_ms", 5.0),
                fade_out_percent=self.params.get("fade_out_percent", 20.0),
                overwrite=self.params.get("overwrite", True),
                stats=stats,
                progress_callback=self._on_file_progress,
                log_callback=self._on_log
            )

            if not success:
                self.log_message.emit(f"Failed to process: {wav_file.name}", "ERROR")

            # Finální file progress
            self.progress_file.emit(100)

        # Finální celkový progress
        self.progress_total.emit(100)

        # Emit statistik
        self.log_message.emit("=== Processing Summary ===", "INFO")
        self.log_message.emit(f"Files: {stats.files_processed} successful, {stats.files_failed} failed", "INFO")
        self.log_message.emit(f"Segments: {stats.segments_created} created, {stats.segments_skipped} skipped", "INFO")
        self.log_message.emit(
            f"Duration: {stats.total_input_duration:.1f}s input → {stats.total_output_duration:.1f}s output",
            "INFO"
        )

        for format_name, count in stats.audio_formats.items():
            self.log_message.emit(f"  {format_name}: {count} files", "INFO")

        self.log_message.emit("Processing completed!", "INFO")

        # Emit finished s statistikami
        self.finished.emit(stats.to_dict())

    def _on_file_progress(self, percent: float):
        """
        Callback pro progress aktuálního souboru.
        Volán z process_wav_file().
        """
        self.progress_file.emit(int(percent))

    def _on_log(self, message: str, level: str):
        """
        Callback pro log messages.
        Volán z process_wav_file().
        """
        self.log_message.emit(message, level)

    def cancel(self):
        """
        Zruší zpracování (graceful shutdown).
        Nastaví flag, který worker kontroluje mezi soubory.
        """
        self._is_cancelled = True
        self.log_message.emit("Cancelling processing...", "WARNING")

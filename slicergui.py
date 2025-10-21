#!/usr/bin/env python
"""
Audio Sample Slicer GUI
========================
Grafické rozhraní pro zpracování audio vzorků.

Usage:
    python slicergui.py

Features:
    - Session-aware s platformdirs persistence
    - MD5 hash-based file caching
    - Real-time progress tracking
    - Asynchronní zpracování (QThread)
    - Auto-save parametrů
"""

import sys
import logging
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox, QDialog

from slicergui.session_manager import SessionManager
from slicergui.gui import SessionDialog, MainWindow


# --- Konfigurace logování ---
def setup_logging():
    """
    Nastavení logovacího systému.
    Loguje do konzole i do souboru.
    """
    log_dir = Path.home() / ".audioslicer"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "slicergui.log"

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formát
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file: {e}")

    return logger


def main():
    """
    Hlavní entry point aplikace.

    Workflow:
        1. Setup logging
        2. Inicializace SessionManager
        3. Zobrazení SessionDialog
        4. Spuštění MainWindow
    """
    # Logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Starting Audio Sample Slicer GUI")
    logger.info("=" * 60)

    # QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("Audio Sample Slicer")
    app.setOrganizationName("LordAudio")

    try:
        # Inicializace session manageru
        session_manager = SessionManager()
        logger.info("SessionManager initialized")

        # Session dialog
        session_dialog = SessionDialog(session_manager)
        if session_dialog.exec() != QDialog.Accepted:
            logger.info("User cancelled session selection - exiting")
            return 0

        selected_session = session_dialog.get_selected_session()
        if not selected_session:
            QMessageBox.critical(None, "Error", "No session selected")
            return 1

        # Load session (pokud již není načtená z create)
        if not session_manager.is_session_loaded():
            if not session_manager.load_session(selected_session):
                QMessageBox.critical(None, "Error", f"Failed to load session: {selected_session}")
                return 1

        logger.info(f"Session loaded: {selected_session}")

        # Hlavní okno
        main_window = MainWindow(session_manager)
        main_window.show()

        logger.info("Main window displayed")

        # Event loop
        exit_code = app.exec()

        logger.info(f"Application exiting with code {exit_code}")
        return exit_code

    except Exception as e:
        logger.exception("Fatal error during startup")
        QMessageBox.critical(
            None,
            "Fatal Error",
            f"An unexpected error occurred:\n\n{str(e)}\n\nSee log file for details."
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

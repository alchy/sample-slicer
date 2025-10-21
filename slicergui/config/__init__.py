"""
Konfigurace aplikace
====================
Platform-aware persistence cesty pomocí platformdirs.
"""

from platformdirs import user_data_dir
from pathlib import Path

# OS-specific session storage
# Windows: C:\Users\{user}\AppData\Local\LordAudio\AudioSlicerSessions
# macOS: ~/Library/Application Support/LordAudio/AudioSlicerSessions
# Linux: ~/.local/share/LordAudio/AudioSlicerSessions
SESSIONS_DIR = Path(user_data_dir(
    appname="AudioSlicerSessions",
    appauthor="LordAudio",
    ensure_exists=True
))

__all__ = ['SESSIONS_DIR']

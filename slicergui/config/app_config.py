"""
Application configuration
=========================
Centralizované konstanty pro GUI a zpracování.
"""

from typing import List


class SessionConfig:
    """Konfigurace session managementu"""

    SESSION_DIR_NAME = "sessions"
    SESSION_FILE_EXTENSION = ".json"
    SESSION_PREFIX = "session-"

    # Nepovolené znaky v názvech sessions
    INVALID_CHARS: List[str] = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']

    MAX_SESSION_NAME_LENGTH = 50
    MIN_SESSION_NAME_LENGTH = 1


class ProcessingDefaults:
    """Výchozí hodnoty pro zpracování audio"""

    # Detekce segmentů
    THRESHOLD_DB = -45.0
    THRESHOLD_MIN = -60.0
    THRESHOLD_MAX = 0.0

    MIN_LENGTH = 3.0
    MIN_LENGTH_MIN = 0.1
    MIN_LENGTH_MAX = 10.0

    MIN_LENGTH_AFTER_TRIM = 0.5
    MIN_LENGTH_AFTER_TRIM_MIN = 0.1
    MIN_LENGTH_AFTER_TRIM_MAX = 5.0

    # Zpracování
    TRIM_THRESHOLD_OFFSET = 10.0
    TRIM_THRESHOLD_OFFSET_MIN = 0.0
    TRIM_THRESHOLD_OFFSET_MAX = 20.0

    # Fade-in (v milisekundách)
    FADE_IN_MS = 5.0
    FADE_IN_MS_MIN = 0.0
    FADE_IN_MS_MAX = 50.0

    # Fade-out (v procentech délky segmentu)
    FADE_OUT_PERCENT = 20.0
    FADE_OUT_PERCENT_MIN = 0.0
    FADE_OUT_PERCENT_MAX = 100.0

    # Flags
    APPLY_FADES = True
    OVERWRITE = True  # Přepisovat existující soubory (default: ano)
    RESUME = False
    PREVIEW = False

    # Logging
    LOG_LEVEL = "INFO"
    LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR"]


class GUIConfig:
    """Konfigurace GUI"""

    # Window dimensions
    WINDOW_WIDTH = 900
    WINDOW_HEIGHT = 750
    WINDOW_MIN_WIDTH = 700
    WINDOW_MIN_HEIGHT = 600

    # Session dialog
    SESSION_DIALOG_WIDTH = 600
    SESSION_DIALOG_HEIGHT = 400

    # Slider precision
    SLIDER_STEPS_THRESHOLD = 600  # -60 až 0 dB (kroky po 0.1)
    SLIDER_STEPS_LENGTH = 99      # 0.1 až 10s (kroky po 0.1)
    SLIDER_STEPS_TRIM = 49        # 0.1 až 5s (kroky po 0.1)
    SLIDER_STEPS_OFFSET = 200     # 0 až 20 dB (kroky po 0.1)
    SLIDER_STEPS_FADE_IN = 500    # 0 až 50 ms (kroky po 0.1)
    SLIDER_STEPS_FADE_OUT = 1000  # 0 až 100 % (kroky po 0.1)

    # Log output
    LOG_MAX_LINES = 1000

    # Folder dialog
    DEFAULT_INPUT_DIR = ""
    DEFAULT_OUTPUT_DIR = ""


class CacheConfig:
    """Konfigurace hash cache"""

    # Hash algoritmus
    HASH_ALGORITHM = "md5"
    CHUNK_SIZE = 8192  # 8KB chunks pro čtení souborů

    # Cache keys
    CACHE_KEY_FILENAME = "filename"
    CACHE_KEY_LAST_PROCESSED = "last_processed"
    CACHE_KEY_SEGMENTS_CREATED = "segments_created"
    CACHE_KEY_AUDIO_FORMAT = "audio_format"
    CACHE_KEY_DURATION = "duration"

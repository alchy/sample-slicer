"""
JSON Session Repository Implementation
=======================================
Konkrétní implementace ISessionRepository s JSON persistence.
Zahrnuje automatický backup a recovery mechanismus.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from slicergui.domain.interfaces.session_repository import ISessionRepository
from slicergui.config import SESSIONS_DIR
from slicergui.config.app_config import SessionConfig, ProcessingDefaults

logger = logging.getLogger(__name__)


class JsonSessionRepository(ISessionRepository):
    """
    Repository pro JSON-based session storage.
    Automaticky vytváří backup před každým save.
    """

    def __init__(self, sessions_folder: Path = None):
        """
        Args:
            sessions_folder: Vlastní složka pro sessions (výchozí: SESSIONS_DIR)
        """
        self.sessions_folder = sessions_folder or SESSIONS_DIR
        self.sessions_folder.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_name: str) -> Path:
        """Vrátí cestu k session souboru"""
        filename = f"{SessionConfig.SESSION_PREFIX}{session_name}{SessionConfig.SESSION_FILE_EXTENSION}"
        return self.sessions_folder / filename

    def _get_backup_path(self, session_path: Path) -> Path:
        """Vrátí cestu k backup souboru"""
        return session_path.with_suffix(f"{SessionConfig.SESSION_FILE_EXTENSION}.backup")

    def create(self, session_name: str) -> Dict[str, Any]:
        """Vytvoří novou session s výchozími hodnotami"""
        if self.exists(session_name):
            raise ValueError(f"Session '{session_name}' already exists")

        # Výchozí session data
        session_data = {
            "session_name": session_name,
            "created": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat(),

            "folders": {
                "input": None,
                "output": None
            },

            "processing_params": {
                "threshold_db": ProcessingDefaults.THRESHOLD_DB,
                "min_length": ProcessingDefaults.MIN_LENGTH,
                "min_length_after_trim": ProcessingDefaults.MIN_LENGTH_AFTER_TRIM,
                "trim_threshold_offset": ProcessingDefaults.TRIM_THRESHOLD_OFFSET,
                "fade_in_ms": ProcessingDefaults.FADE_IN_MS,
                "fade_out_percent": ProcessingDefaults.FADE_OUT_PERCENT,
                "apply_fades": ProcessingDefaults.APPLY_FADES,
                "overwrite": ProcessingDefaults.OVERWRITE,
                "resume": ProcessingDefaults.RESUME,
                "preview": ProcessingDefaults.PREVIEW,
                "log_level": ProcessingDefaults.LOG_LEVEL
            },

            "files_cache": {},

            "settings": {
                "ui_state": {}
            }
        }

        # Uložení nové session
        if self.save(session_name, session_data):
            logger.info(f"Created new session: {session_name}")
            return session_data
        else:
            raise RuntimeError(f"Failed to create session: {session_name}")

    def load(self, session_name: str) -> Optional[Dict[str, Any]]:
        """Načte session ze souboru"""
        session_path = self._get_session_path(session_name)

        if not session_path.exists():
            logger.warning(f"Session not found: {session_name}")
            return None

        try:
            with open(session_path, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            logger.info(f"Loaded session: {session_name}")
            return session_data

        except json.JSONDecodeError as e:
            logger.error(f"Corrupted session file: {session_name} - {e}")

            # Pokus o obnovu z backup
            backup_path = self._get_backup_path(session_path)
            if backup_path.exists():
                logger.info("Attempting to restore from backup...")
                try:
                    with open(backup_path, 'r', encoding='utf-8') as f:
                        session_data = json.load(f)
                    logger.info("Successfully restored from backup")
                    return session_data
                except Exception as backup_error:
                    logger.error(f"Backup also corrupted: {backup_error}")

            return None

        except Exception as e:
            logger.error(f"Error loading session {session_name}: {e}")
            return None

    def save(self, session_name: str, session_data: Dict[str, Any]) -> bool:
        """Uloží session s automatickým backupem"""
        session_path = self._get_session_path(session_name)

        try:
            # Backup existujícího souboru před přepsáním
            if session_path.exists():
                backup_path = self._get_backup_path(session_path)
                try:
                    session_path.replace(backup_path)
                except Exception as e:
                    logger.warning(f"Failed to create backup: {e}")

            # Update last_modified timestamp
            session_data["last_modified"] = datetime.now().isoformat()

            # Uložení nových dat
            with open(session_path, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved session: {session_name}")
            return True

        except Exception as e:
            logger.error(f"Error saving session {session_name}: {e}")

            # Pokus o obnovu z backup při chybě
            backup_path = self._get_backup_path(session_path)
            if backup_path.exists():
                try:
                    backup_path.replace(session_path)
                    logger.info("Restored from backup after save failure")
                except Exception as restore_error:
                    logger.error(f"Failed to restore backup: {restore_error}")

            return False

    def delete(self, session_name: str) -> bool:
        """Smaže session i její backup"""
        session_path = self._get_session_path(session_name)
        backup_path = self._get_backup_path(session_path)

        success = True

        try:
            if session_path.exists():
                session_path.unlink()
                logger.info(f"Deleted session: {session_name}")
        except Exception as e:
            logger.error(f"Error deleting session {session_name}: {e}")
            success = False

        try:
            if backup_path.exists():
                backup_path.unlink()
        except Exception as e:
            logger.warning(f"Error deleting backup for {session_name}: {e}")

        return success

    def list_sessions(self) -> List[str]:
        """Vrátí seznam všech sessions"""
        pattern = f"{SessionConfig.SESSION_PREFIX}*{SessionConfig.SESSION_FILE_EXTENSION}"

        sessions = []
        for session_file in self.sessions_folder.glob(pattern):
            # Ignoruj backup soubory
            if session_file.suffix == ".backup":
                continue

            # Extrahuj název session (odstraň prefix a suffix)
            name = session_file.stem
            if name.startswith(SessionConfig.SESSION_PREFIX):
                name = name[len(SessionConfig.SESSION_PREFIX):]
            sessions.append(name)

        sessions.sort()
        return sessions

    def exists(self, session_name: str) -> bool:
        """Zkontroluje existenci session"""
        return self._get_session_path(session_name).exists()

"""
Session Manager
===============
High-level API pro správu sessions včetně MD5 hash caching.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from slicergui.infrastructure.persistence.session_repository_impl import JsonSessionRepository
from slicergui.config.app_config import CacheConfig

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Správa session s hash-based cache pro zpracované soubory.
    """

    def __init__(self, repository: JsonSessionRepository = None):
        """
        Args:
            repository: Custom repository (výchozí: JsonSessionRepository)
        """
        self.repository = repository or JsonSessionRepository()
        self.current_session_name: Optional[str] = None
        self.session_data: Optional[Dict[str, Any]] = None

    # --- Session Operations ---

    def create_session(self, session_name: str) -> bool:
        """
        Vytvoří novou session a nastaví ji jako aktivní.

        Returns:
            True pokud úspěch
        """
        try:
            self.session_data = self.repository.create(session_name)
            self.current_session_name = session_name
            logger.info(f"Created and loaded session: {session_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to create session {session_name}: {e}")
            return False

    def load_session(self, session_name: str) -> bool:
        """
        Načte existující session.

        Returns:
            True pokud úspěch
        """
        session_data = self.repository.load(session_name)
        if session_data:
            self.session_data = session_data
            self.current_session_name = session_name
            logger.info(f"Loaded session: {session_name}")
            return True
        else:
            logger.error(f"Failed to load session: {session_name}")
            return False

    def save_session(self) -> bool:
        """
        Uloží aktuální session.

        Returns:
            True pokud úspěch
        """
        if not self.current_session_name or not self.session_data:
            logger.warning("No active session to save")
            return False

        return self.repository.save(self.current_session_name, self.session_data)

    def delete_session(self, session_name: str) -> bool:
        """
        Smaže session.

        Returns:
            True pokud úspěch
        """
        success = self.repository.delete(session_name)

        # Pokud mazaná session je aktivní, vyčisti state
        if success and session_name == self.current_session_name:
            self.current_session_name = None
            self.session_data = None

        return success

    def list_all_sessions(self) -> List[str]:
        """Vrátí seznam všech sessions"""
        return self.repository.list_sessions()

    def is_session_loaded(self) -> bool:
        """Zkontroluje, zda je nějaká session načtená"""
        return self.session_data is not None

    # --- Parameters Management ---

    def get_folders(self) -> Dict[str, Optional[str]]:
        """Vrátí input/output složky"""
        if not self.session_data:
            return {"input": None, "output": None}
        return self.session_data.get("folders", {"input": None, "output": None})

    def set_folders(self, input_dir: Optional[str] = None, output_dir: Optional[str] = None):
        """Nastaví input/output složky"""
        if not self.session_data:
            return

        if input_dir is not None:
            self.session_data["folders"]["input"] = input_dir
        if output_dir is not None:
            self.session_data["folders"]["output"] = output_dir

        self.save_session()

    def get_processing_params(self) -> Dict[str, Any]:
        """Vrátí všechny zpracovací parametry"""
        if not self.session_data:
            return {}
        return self.session_data.get("processing_params", {})

    def update_processing_params(self, **kwargs):
        """
        Aktualizuje zpracovací parametry.

        Example:
            manager.update_processing_params(threshold_db=-50, min_length=2.5)
        """
        if not self.session_data:
            return

        params = self.session_data.get("processing_params", {})
        params.update(kwargs)
        self.session_data["processing_params"] = params

        self.save_session()

    def get_ui_state(self) -> Dict[str, Any]:
        """Vrátí UI state"""
        if not self.session_data:
            return {}
        return self.session_data.get("settings", {}).get("ui_state", {})

    def update_ui_state(self, **kwargs):
        """Aktualizuje UI state"""
        if not self.session_data:
            return

        if "settings" not in self.session_data:
            self.session_data["settings"] = {}
        if "ui_state" not in self.session_data["settings"]:
            self.session_data["settings"]["ui_state"] = {}

        self.session_data["settings"]["ui_state"].update(kwargs)
        self.save_session()

    # --- Hash Cache Management ---

    def calculate_file_hash(self, file_path: Path) -> str:
        """
        Spočítá MD5 hash souboru.

        Args:
            file_path: Cesta k souboru

        Returns:
            Hex string MD5 hash
        """
        hash_md5 = hashlib.md5()

        try:
            with open(file_path, "rb") as f:
                while chunk := f.read(CacheConfig.CHUNK_SIZE):
                    hash_md5.update(chunk)
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {e}")
            return ""

        return hash_md5.hexdigest()

    def is_file_cached(self, file_hash: str) -> bool:
        """
        Zkontroluje, zda je soubor v cache.

        Args:
            file_hash: MD5 hash souboru

        Returns:
            True pokud je v cache
        """
        if not self.session_data:
            return False

        cache = self.session_data.get("files_cache", {})
        return file_hash in cache

    def get_cached_file_data(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Vrátí cached data pro soubor.

        Args:
            file_hash: MD5 hash

        Returns:
            Dict s cached data nebo None
        """
        if not self.session_data:
            return None

        cache = self.session_data.get("files_cache", {})
        return cache.get(file_hash)

    def add_file_to_cache(
        self,
        file_path: Path,
        segments_created: int,
        audio_format: str,
        duration: float
    ) -> str:
        """
        Přidá soubor do cache.

        Args:
            file_path: Cesta k souboru
            segments_created: Počet vytvořených segmentů
            audio_format: Formát (např. "48000Hz_2ch_16bit")
            duration: Délka v sekundách

        Returns:
            Hash souboru
        """
        if not self.session_data:
            return ""

        file_hash = self.calculate_file_hash(file_path)
        if not file_hash:
            return ""

        cache_entry = {
            CacheConfig.CACHE_KEY_FILENAME: file_path.name,
            CacheConfig.CACHE_KEY_LAST_PROCESSED: datetime.now().isoformat(),
            CacheConfig.CACHE_KEY_SEGMENTS_CREATED: segments_created,
            CacheConfig.CACHE_KEY_AUDIO_FORMAT: audio_format,
            CacheConfig.CACHE_KEY_DURATION: duration
        }

        if "files_cache" not in self.session_data:
            self.session_data["files_cache"] = {}

        self.session_data["files_cache"][file_hash] = cache_entry
        self.save_session()

        logger.debug(f"Added to cache: {file_path.name} (hash: {file_hash[:8]}...)")
        return file_hash

    def clear_cache(self):
        """Vymaže celou cache"""
        if not self.session_data:
            return

        self.session_data["files_cache"] = {}
        self.save_session()
        logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Vrátí statistiky cache.

        Returns:
            Dict s total_files, total_segments, etc.
        """
        if not self.session_data:
            return {"total_files": 0, "total_segments": 0}

        cache = self.session_data.get("files_cache", {})

        total_files = len(cache)
        total_segments = sum(
            entry.get(CacheConfig.CACHE_KEY_SEGMENTS_CREATED, 0)
            for entry in cache.values()
        )

        return {
            "total_files": total_files,
            "total_segments": total_segments
        }

    # --- Session Info ---

    def get_session_info(self) -> Dict[str, Any]:
        """
        Vrátí základní info o session.

        Returns:
            Dict s name, created, last_modified
        """
        if not self.session_data:
            return {}

        return {
            "session_name": self.session_data.get("session_name", ""),
            "created": self.session_data.get("created", ""),
            "last_modified": self.session_data.get("last_modified", "")
        }

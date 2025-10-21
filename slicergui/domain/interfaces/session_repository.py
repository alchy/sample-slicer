"""
Session Repository Interface
=============================
Abstraktní rozhraní pro session persistence - Repository pattern.
Umožňuje snadnou výměnu implementace (JSON → SQL → Redis).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class ISessionRepository(ABC):
    """
    Interface pro session persistence.
    Implementace musí poskytovat CRUD operace + listing.
    """

    @abstractmethod
    def create(self, session_name: str) -> Dict[str, Any]:
        """
        Vytvoří novou session s výchozími hodnotami.

        Args:
            session_name: Název session (bez přípony)

        Returns:
            Dict s session data

        Raises:
            ValueError: Session již existuje
        """
        pass

    @abstractmethod
    def load(self, session_name: str) -> Optional[Dict[str, Any]]:
        """
        Načte existující session.

        Args:
            session_name: Název session

        Returns:
            Dict s session data nebo None pokud neexistuje
        """
        pass

    @abstractmethod
    def save(self, session_name: str, session_data: Dict[str, Any]) -> bool:
        """
        Uloží session data (s automatickým backupem).

        Args:
            session_name: Název session
            session_data: Data k uložení

        Returns:
            True pokud úspěch, False jinak
        """
        pass

    @abstractmethod
    def delete(self, session_name: str) -> bool:
        """
        Smaže session.

        Args:
            session_name: Název session

        Returns:
            True pokud úspěch, False jinak
        """
        pass

    @abstractmethod
    def list_sessions(self) -> List[str]:
        """
        Vrátí seznam všech dostupných sessions.

        Returns:
            List názvů sessions (bez přípony)
        """
        pass

    @abstractmethod
    def exists(self, session_name: str) -> bool:
        """
        Zkontroluje, zda session existuje.

        Args:
            session_name: Název session

        Returns:
            True pokud existuje
        """
        pass

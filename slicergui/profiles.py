"""Profily GUI: pojmenované sady složek a parametrů v jednom JSON souboru (bez Qt, testovatelné)."""
from __future__ import annotations
import json
from pathlib import Path

DEFAULT_PARAMS = {
    "mode": "build",            # "build" = banka pro ithaca, "slice" = generický střih
    "src": "", "original": "", "bank": "", "out": "",
    "end_level_db": -60.0, "tail_s": 2.0, "max_len_s": 30.0,
    "preroll_ms": 5.0, "fade_in_ms": 2.0, "retune": False,
}


def default_store_path() -> Path:
    from platformdirs import user_config_dir
    return Path(user_config_dir("sample-slicer", "alchy")) / "profiles.json"


class ProfileStore:
    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else default_store_path()
        self.data = {"last": None, "profiles": {}}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text())
            except (OSError, ValueError):
                pass                       # rozbitý soubor = začni od nuly, ale nepřepisuj dokud se neuloží
        self.data.setdefault("last", None)
        self.data.setdefault("profiles", {})

    def names(self) -> list[str]:
        return sorted(self.data["profiles"])

    def get(self, name: str) -> dict:
        p = dict(DEFAULT_PARAMS)
        p.update(self.data["profiles"].get(name, {}))
        return p

    def put(self, name: str, params: dict) -> None:
        name = name.strip()
        if not name:
            raise ValueError("název profilu je prázdný")
        self.data["profiles"][name] = {k: params.get(k, v) for k, v in DEFAULT_PARAMS.items()}
        self.data["last"] = name
        self._save()

    def delete(self, name: str) -> None:
        self.data["profiles"].pop(name, None)
        if self.data["last"] == name:
            self.data["last"] = None
        self._save()

    @property
    def last(self) -> str | None:
        return self.data["last"] if self.data["last"] in self.data["profiles"] else None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))

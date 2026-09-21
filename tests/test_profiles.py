import json, pytest
from slicergui.profiles import ProfileStore, DEFAULT_PARAMS

def test_roundtrip_and_last(tmp_path):
    s = ProfileStore(tmp_path / "p.json")
    assert s.names() == [] and s.last is None
    s.put("petrof", {"src": "/raw", "end_level_db": -50, "mode": "build"})
    s2 = ProfileStore(tmp_path / "p.json")
    assert s2.names() == ["petrof"] and s2.last == "petrof"
    p = s2.get("petrof")
    assert p["src"] == "/raw" and p["end_level_db"] == -50 and p["tail_s"] == DEFAULT_PARAMS["tail_s"]
    s2.delete("petrof")
    assert ProfileStore(tmp_path / "p.json").names() == []

def test_unknown_keys_dropped_and_empty_name_rejected(tmp_path):
    s = ProfileStore(tmp_path / "p.json")
    s.put("x", {"bogus": 1})
    assert "bogus" not in json.loads((tmp_path / "p.json").read_text())["profiles"]["x"]
    with pytest.raises(ValueError):
        s.put("  ", {})

def test_corrupt_file_starts_empty(tmp_path):
    (tmp_path / "p.json").write_text("{not json")
    assert ProfileStore(tmp_path / "p.json").names() == []

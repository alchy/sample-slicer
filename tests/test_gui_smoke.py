import os, pytest
pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from slicergui.app import MainWindow
from slicergui.profiles import ProfileStore

@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])

def test_window_params_roundtrip_and_mode_visibility(app, tmp_path):
    w = MainWindow(ProfileStore(tmp_path / "p.json"))
    p = w.params()
    assert p["mode"] == "build" and p["end_level_db"] == -60.0
    assert w.dir_edits["bank"][1].isVisibleTo(w) and not w.dir_edits["out"][1].isVisibleTo(w)
    w.set_params({**p, "mode": "slice", "src": "/a", "out": "/b", "end_level_db": -50.0, "retune": True})
    q = w.params()
    assert q["mode"] == "slice" and q["src"] == "/a" and q["out"] == "/b" and q["end_level_db"] == -50.0
    assert w.dir_edits["out"][1].isVisibleTo(w) and not w.dir_edits["bank"][1].isVisibleTo(w)
    w.store.put("t", q)
    assert ProfileStore(tmp_path / "p.json").get("t")["out"] == "/b"

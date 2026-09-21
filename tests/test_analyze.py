import json, numpy as np
from sample_slicer.io import write_wav
from sample_slicer.analyze import analyze_dir, accuracy_by_octave
from sample_slicer.detect import DetectParams
from sample_slicer.cli import main
from tests.synth import piano_note, place, noise_floor

SR = 96000

def make_src(tmp_path, midis=(60, 62, 64)):
    src = tmp_path / "src"; src.mkdir()
    c = noise_floor(int((2 + 3 * len(midis)) * SR), SR, db=-85)
    for i, m in enumerate(midis):
        place(c, piano_note(m, SR, 2.0, decay_db_s=-20, inharmonicity=0.0003), 1.0 + 3.0 * i, SR)
    write_wav(src / "rec.wav", c[:, None], SR, 24)
    (src / "truth.json").write_text(json.dumps({"rec.wav": {"start": "C4", "pattern": "major"}}))
    return src

def test_analyze_with_truth(tmp_path):
    src = make_src(tmp_path)
    truth = json.loads((src / "truth.json").read_text())
    rows = analyze_dir(src, DetectParams(), 2.0, truth, log=lambda *_: None)
    assert [r.assignment.midi for r in rows] == [60, 62, 64]
    assert [r.truth for r in rows] == [60, 62, 64]
    acc = accuracy_by_octave(rows)
    assert sum(v["ok"] for v in acc.values()) == 3 and sum(v["octave"] + v["other"] for v in acc.values()) == 0

def test_cli_analyze_prints_and_writes_nothing(tmp_path, capsys):
    src = make_src(tmp_path)
    before = sorted(p.name for p in src.iterdir())
    assert main(["analyze", str(src), "--truth", str(src / "truth.json")]) == 0
    out = capsys.readouterr().out
    assert "C4" in out and "ok" in out
    assert sorted(p.name for p in src.iterdir()) == before

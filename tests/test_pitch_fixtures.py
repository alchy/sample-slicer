import json
from pathlib import Path
import pytest
from sample_slicer.io import read_wav
from sample_slicer.pitch import estimate_pitch

FIX = Path(__file__).parent / "fixtures" / "pitch"
TRUTH = json.loads((FIX / "truth.json").read_text())

@pytest.mark.parametrize("name", sorted(TRUTH))
def test_fixture_note(name):
    data, info = read_wav(FIX / name)
    p = estimate_pitch(data[:, 0], info.sample_rate)
    assert p is not None
    truth = TRUTH[name]
    diff = p.midi - truth
    # laťka detektoru: správná oktáva a do ±90 c (rozladěné B7/C8 řeší ladicí křivka)
    assert abs(diff) < 0.9, f"{name}: midi {p.midi:.2f} vs {truth} ({100*diff:+.0f} c), conf {p.confidence:.2f}"

def test_no_octave_errors_overall():
    bad = []
    for name, truth in TRUTH.items():
        data, info = read_wav(FIX / name)
        p = estimate_pitch(data[:, 0], info.sample_rate)
        if p is None or abs(p.midi - truth) >= 0.9:
            bad.append((name, None if p is None else round(p.midi, 2), truth))
    assert bad == []

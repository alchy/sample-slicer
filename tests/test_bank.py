import json, shutil, numpy as np, pytest
from pathlib import Path
from sample_slicer.io import write_wav, read_wav
from sample_slicer.bank import build_bank, md5_16, ffmpeg_available, BankIndex
from sample_slicer.detect import DetectParams
from sample_slicer.tuning import TuningParams
from tests.synth import piano_note, place, noise_floor

SR = 96000
needs_ffmpeg = pytest.mark.skipif(not ffmpeg_available(), reason="ffmpeg chybí")

def make_src(tmp_path, midis=(60, 62), name="rec.wav"):
    src = tmp_path / "src"; src.mkdir(exist_ok=True)
    c = noise_floor(int((2 + 3 * len(midis)) * SR), SR, db=-85)
    for i, m in enumerate(midis):
        place(c, piano_note(m, SR, 2.0, decay_db_s=-20, inharmonicity=0.0003), 1.0 + 3.0 * i, SR)
    write_wav(src / name, np.stack([c, c], axis=1), SR, 24)
    return src

def run(src, tmp_path, **kw):
    return build_bank(src, tmp_path / "orig", tmp_path / "out", DetectParams(), 2.0, TuningParams(),
                      log=lambda *_: None, **kw)

@needs_ffmpeg
def test_build_layout_and_formats(tmp_path):
    src = make_src(tmp_path)
    res = run(src, tmp_path)
    assert res.written == 2 and res.rejected == 0
    orig = sorted((tmp_path / "orig").glob("m*/*.wav")); out = sorted((tmp_path / "out").glob("m*/*.wav"))
    assert [p.parent.name for p in orig] == ["m060", "m062"] and len(out) == 2
    assert all(len(p.stem) == 16 for p in orig + out)
    assert orig[0].stem == md5_16(orig[0])
    d, info = read_wav(orig[0]); assert info.sample_rate == SR and info.bits == 24 and info.channels == 2
    d, info = read_wav(out[0]); assert info.sample_rate == 48000 and info.bits == 16 and info.channels == 2
    assert (tmp_path / "orig" / "report.md").exists() and (tmp_path / "orig" / ".slicer-index.json").exists()

@needs_ffmpeg
def test_build_is_idempotent_and_keeps_foreign_files(tmp_path):
    src = make_src(tmp_path)
    run(src, tmp_path)
    foreign = tmp_path / "orig" / "m060" / "manual.wav"; foreign.write_bytes(b"x")
    snapshot = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.wav"))
    res = run(src, tmp_path)
    assert res.skipped_sources == 1 and res.written == 0
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*.wav")) == snapshot
    assert foreign.exists()

@needs_ffmpeg
def test_changed_params_replace_old_files(tmp_path):
    src = make_src(tmp_path)
    run(src, tmp_path)
    old = sorted((tmp_path / "orig").glob("m*/*.wav"))
    res = build_bank(src, tmp_path / "orig", tmp_path / "out", DetectParams(), 1.0, TuningParams(), log=lambda *_: None)
    new = sorted((tmp_path / "orig").glob("m*/*.wav"))
    assert res.written == 2 and len(new) == 2 and set(new).isdisjoint(set(old))

@needs_ffmpeg
def test_overrides_skip_and_midi(tmp_path):
    src = make_src(tmp_path, midis=(60, 62, 64))
    res = run(src, tmp_path, overrides={"rec.wav": {"skip": [1], "midi": {"2": 70}}})
    assert res.written == 2
    assert sorted(p.name for p in (tmp_path / "orig").glob("m*")) == ["m060", "m070"]
    assert "override" in (tmp_path / "orig" / "report.md").read_text()

@needs_ffmpeg
def test_rejected_goes_to_folder(tmp_path):
    src = make_src(tmp_path, midis=(60,))
    # druhý „úder" = čistý šum s hlasitostí tónu → nízká confidence → _rejected
    data, info = read_wav(src / "rec.wav")
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal((SR, 2)) * 0.1).astype(np.float32)
    write_wav(src / "rec.wav", np.concatenate([data, noise, np.zeros((SR, 2), np.float32)]), SR, 24)
    res = run(src, tmp_path)
    assert res.rejected >= 1
    assert list((tmp_path / "orig" / "_rejected").glob("rec_*.wav"))

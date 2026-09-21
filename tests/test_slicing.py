import numpy as np
from sample_slicer.io import write_wav, read_wav
from sample_slicer.slicing import slice_file, slice_dir
from sample_slicer.cli import main
from tests.synth import piano_note, place, noise_floor

SR = 48000

def make_file(path, notes=(60, 64), bits=24):
    c = noise_floor(int(10 * SR), SR, db=-85)
    for i, m in enumerate(notes):
        place(c, piano_note(m, SR, 3.0, decay_db_s=-15), 1.0 + 4.0 * i, SR)
    write_wav(path, np.stack([c, 0.8 * c], axis=1), SR, bits)

def test_slice_file_returns_stereo_slices(tmp_path):
    p = tmp_path / "in.wav"; make_file(p)
    slices, rejected = slice_file(p)
    assert len(slices) == 2 and rejected == []
    s = slices[0]
    assert s.audio.ndim == 2 and s.audio.shape[1] == 2 and s.bits == 24 and s.sr == SR
    assert s.audio[-1, 0] == 0.0                     # ocas do nuly
    assert s.index == 0 and s.source == p

def test_slice_dir_writes_named_files(tmp_path):
    src = tmp_path / "src"; src.mkdir(); make_file(src / "rec.wav")
    out = tmp_path / "out"
    n = slice_dir(src, out)
    files = sorted(out.glob("rec_slice_*.wav"))
    assert n == 2 and len(files) == 2
    assert files[0].name.startswith("rec_slice_001_start_")
    data, info = read_wav(files[0])
    assert info.bits == 24 and info.channels == 2

def test_cli_slice(tmp_path):
    src = tmp_path / "src"; src.mkdir(); make_file(src / "rec.wav")
    out = tmp_path / "out"
    assert main(["slice", str(src), str(out)]) == 0
    assert len(list(out.glob("*.wav"))) == 2

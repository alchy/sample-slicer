import numpy as np
from tests.synth import piano_note, thump, place, noise_floor

def test_piano_note_shape_and_peak():
    x = piano_note(69, 48000, 1.0, peak=0.5)
    assert x.shape == (48000,) and x.dtype == np.float32
    assert 0.45 < np.abs(x).max() <= 0.5

def test_piano_note_decays():
    x = piano_note(69, 48000, 2.0, decay_db_s=-20.0)
    a = np.sqrt(np.mean(x[4800:9600] ** 2)); b = np.sqrt(np.mean(x[48000:52800] ** 2))
    assert 20 * np.log10(b / a) < -12   # za ~0.9 s aspoň 12 dB dolů

def test_place_and_floor():
    c = noise_floor(48000, 48000, db=-80)
    place(c, np.ones(10, dtype=np.float32), 0.5, 48000)
    assert c[24000] > 0.9 and abs(c[100]) < 1e-3
    assert len(thump(48000)) == 1440

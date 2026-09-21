import numpy as np, pytest
from sample_slicer.pitch import estimate_pitch, midi_from_hz, hz_from_midi
from tests.synth import piano_note, thump, place, noise_floor

SR = 96000

def note_with_context(midi, dur=1.0, **kw):
    x = noise_floor(int(dur * SR), SR, db=-80)
    x += piano_note(midi, SR, dur, **kw)
    return x

@pytest.mark.parametrize("midi", [21, 24, 33, 45, 60, 72, 84, 96, 105, 108])
def test_synthetic_notes_across_keyboard(midi):
    x = note_with_context(midi, inharmonicity=0.0005 if midi > 60 else 0.0001, decay_db_s=-8 if midi < 60 else -30)
    p = estimate_pitch(x, SR)
    assert p is not None
    assert abs(p.midi - midi) < 0.3, (p, midi)
    assert p.evidence >= 2

def test_weak_fundamental_bass():
    # A0 s fundamentálem 30 dB pod parciálami 2–6 (jako u reálného basu)
    dur = 1.5
    t = np.arange(int(dur * SR)) / SR
    f0 = hz_from_midi(21)
    x = np.zeros_like(t)
    for h, amp in [(1, 0.03), (2, 0.6), (3, 0.8), (4, 1.0), (5, 0.7), (6, 0.5)]:
        x += amp * np.exp(-2 * t) * np.sin(2 * np.pi * h * f0 * np.sqrt(1 + 1e-4 * h * h) * t)
    x = (0.4 * x / np.abs(x).max()).astype(np.float32) + noise_floor(len(t), SR, db=-80)
    p = estimate_pitch(x, SR)
    assert abs(p.midi - 21) < 0.3, p

def test_thump_does_not_win_on_short_treble():
    x = note_with_context(100, dur=0.7, decay_db_s=-40, inharmonicity=0.001)
    place(x, thump(SR, peak=0.3), 0.0, SR)       # silný úder kladívka na začátku
    p = estimate_pitch(x, SR)
    assert abs(p.midi - 100) < 0.3, p

def test_conversions():
    assert abs(midi_from_hz(440) - 69) < 1e-9 and abs(hz_from_midi(21) - 27.5) < 1e-9

def test_silence_returns_none():
    assert estimate_pitch(np.zeros(SR, dtype=np.float32), SR) is None

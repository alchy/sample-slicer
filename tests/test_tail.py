import numpy as np
from sample_slicer.tail import apply_fade_in, natural_tail, render_segment
from sample_slicer.detect import Segment
from sample_slicer.envelope import rms_envelope_db

SR = 48000

def test_fade_in_starts_at_zero():
    x = np.ones((SR, 1), dtype=np.float32)
    y = apply_fade_in(x, SR, ms=2.0)
    assert y[0, 0] == 0.0 and y[95, 0] < 1.0 and y[96, 0] == 1.0

def test_natural_tail_monotone_and_ends_zero():
    x = np.ones((4 * SR, 1), dtype=np.float32)
    y = natural_tail(x, SR, slope_db_s=-3.0, tail_s=2.0)
    assert np.array_equal(y[: 2 * SR], x[: 2 * SR])          # před ocasem beze změny
    env = rms_envelope_db(y[:, 0], SR, hop_ms=50)
    tail_env = env[40:]                                        # posledních 2 s
    assert np.all(np.diff(tail_env) <= 1e-6)                   # monotónně klesá
    assert y[-1, 0] == 0.0
    # strmost ≥ požadovaná: za 2 s musí dojít aspoň na -96 dB
    assert env[-2] < -80

def test_natural_tail_uses_measured_slope_when_steeper():
    x = np.ones((3 * SR, 1), dtype=np.float32)
    y = natural_tail(x, SR, slope_db_s=-100.0, tail_s=2.0)     # naměřených -100 dB/s je strmější než 48 dB/s
    env = rms_envelope_db(y[:, 0], SR, hop_ms=50)
    i0 = len(env) - 40                                         # začátek ocasu (2 s = 40 rámců po 50 ms)
    assert env[i0 + 10] < -45                                  # po 0.5 s ocasu ~-50 dB (100 dB/s)
    assert env[i0 + 20] < -95                                  # po 1.0 s ocasu ~-100 dB

def test_render_segment_pads_at_eof():
    audio = np.ones((SR, 2), dtype=np.float32)
    seg = Segment(start=0, onset=100, end=SR - 100, peak_db=0, floor_db=-80, slope_db_s=-3, end_reason="eof")
    out = render_segment(audio, SR, seg, tail_s=1.0)
    assert out.shape == (SR - 100 + SR, 2)
    assert out[-1, 0] == 0.0 and out[0, 0] == 0.0

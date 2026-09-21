import numpy as np
from sample_slicer.envelope import rms_envelope_db, smooth_db, local_floor_db, decay_slope_db_s, hop_frames

SR = 48000

def test_envelope_of_constant_sine():
    t = np.arange(SR) / SR
    x = 0.5 * np.sin(2 * np.pi * 1000 * t)
    env = rms_envelope_db(x, SR, hop_ms=10)
    assert env.shape[0] == 100
    # RMS sinusu 0.5 = 0.3536 → -9.03 dB
    assert np.allclose(env[5:-5], -9.03, atol=0.2)

def test_envelope_silence_floor():
    env = rms_envelope_db(np.zeros(SR), SR)
    assert np.all(env == -120.0)

def test_smooth_median_removes_spike():
    env = np.full(50, -40.0); env[25] = 0.0
    assert smooth_db(env, 5)[25] == -40.0

def test_local_floor():
    hop = 0.005
    env = np.full(1000, -70.0); env[300:400] = -20.0   # tón v 1.5–2.0 s
    # před nasazením v 3.0 s (index 600) je okno 2 s = indexy 200..600, 5. percentil ~ -70
    assert local_floor_db(env, 600, hop) == -70.0
    assert local_floor_db(env, 0, hop) == -70.0

def test_decay_slope():
    hop = 0.005
    t = np.arange(2000) * hop
    env = -20 - 3.0 * t   # -3 dB/s
    assert abs(decay_slope_db_s(env, 2000, hop) - (-3.0)) < 1e-6

def test_hop_frames():
    assert hop_frames(96000, 5.0) == 480

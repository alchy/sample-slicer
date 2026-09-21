import numpy as np, pytest
from sample_slicer.io import read_wav, write_wav, remove_dc, to_mono, UnsupportedWav, WavInfo

@pytest.mark.parametrize("bits", [16, 24, 32])
def test_roundtrip_bits(tmp_path, bits):
    sr = 48000
    t = np.arange(4800) / sr
    data = np.stack([0.5*np.sin(2*np.pi*440*t), -0.25*np.sin(2*np.pi*220*t)], axis=1).astype(np.float32)
    p = tmp_path / f"x{bits}.wav"
    write_wav(p, data, sr, bits)
    back, info = read_wav(p)
    assert info == WavInfo(sample_rate=sr, channels=2, bits=bits, frames=4800)
    assert back.shape == (4800, 2)
    assert np.max(np.abs(back - data)) < 2.0 / (2 ** (bits - 1)) + 1e-6

def test_roundtrip_24bit_bit_exact(tmp_path):
    # 24bit hodnoty přesně reprezentovatelné → po round-tripu identické
    sr = 96000
    q = 2 ** 23
    ints = np.array([[-q, q - 1], [0, 1], [-1, 12345]], dtype=np.int64)
    data = (ints / q).astype(np.float32)
    p = tmp_path / "exact.wav"
    write_wav(p, data, sr, 24)
    back, info = read_wav(p)
    assert info.bits == 24 and info.channels == 2
    assert np.array_equal(np.round(back * q).astype(np.int64), ints)

def test_mono_is_2d(tmp_path):
    data = np.zeros((100, 1), dtype=np.float32)
    p = tmp_path / "m.wav"
    write_wav(p, data, 44100, 16)
    back, info = read_wav(p)
    assert back.shape == (100, 1) and info.channels == 1

def test_remove_dc_and_mono():
    data = np.array([[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]], dtype=np.float32)
    nodc = remove_dc(data)
    assert np.allclose(nodc.mean(axis=0), 0)
    assert np.allclose(to_mono(data), [0.5, 0.75, 1.0])

def test_float_wav_rejected(tmp_path):
    import struct
    p = tmp_path / "f.wav"
    pcm = np.zeros(10, dtype=np.float32).tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, 48000, 48000*4, 4, 32) + b"data" + struct.pack("<I", len(pcm))
    p.write_bytes(hdr + pcm)
    with pytest.raises(UnsupportedWav):
        read_wav(p)

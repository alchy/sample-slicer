import numpy as np
from sample_slicer.detect import detect_segments, DetectParams
from tests.synth import piano_note, thump, place, noise_floor

SR = 48000

def make_canvas(sec):
    return noise_floor(int(sec * SR), SR, db=-85)

def test_two_notes_onsets_within_5ms():
    c = make_canvas(8)
    place(c, piano_note(60, SR, 3.0, decay_db_s=-15), 1.0, SR)
    place(c, piano_note(64, SR, 3.0, decay_db_s=-15), 5.0, SR)
    segs, rej = detect_segments(c, SR)
    assert len(segs) == 2 and rej == []
    assert abs(segs[0].onset / SR - 1.0) < 0.005
    assert abs(segs[1].onset / SR - 5.0) < 0.005
    assert 0 <= segs[0].onset - segs[0].start <= int(0.02 * SR)   # pre-roll ≤ 20 ms

def test_merged_notes_are_split():
    c = make_canvas(6)
    place(c, piano_note(60, SR, 4.0, decay_db_s=-20), 1.0, SR)
    place(c, piano_note(67, SR, 3.0, decay_db_s=-20), 2.0, SR)   # druhý tón do dozvuku prvního (o ~20 dB slabšího)
    segs, _ = detect_segments(c, SR)
    assert len(segs) == 2
    assert segs[0].end_reason == "next_onset" and segs[0].end <= segs[1].start
    assert abs(segs[1].onset / SR - 2.0) < 0.005

def test_end_at_level():
    c = make_canvas(12)
    place(c, piano_note(60, SR, 10.0, peak=0.5, decay_db_s=-10), 1.0, SR)   # obálka ~-12 dB, -40 dB za ~2.7 s
    segs, _ = detect_segments(c, SR, DetectParams(end_level_db=-40.0))
    assert len(segs) == 1 and segs[0].end_reason == "level"
    assert 2.0 < (segs[0].end - segs[0].onset) / SR < 4.0
    assert segs[0].slope_db_s < -5

def test_release_artifact_cuts_before_thump():
    c = make_canvas(10)
    place(c, piano_note(60, SR, 8.0, peak=0.5, decay_db_s=-10), 1.0, SR)   # ve 4 s už ~-45 dB
    place(c, thump(SR, peak=0.5), 4.0, SR)                       # pád kladívka ve 4.0 s, ~-15 dB v rámci
    segs, _ = detect_segments(c, SR, DetectParams(end_level_db=-90.0))
    assert len(segs) == 1 and segs[0].end_reason == "artifact"
    assert 3.7 < segs[0].end / SR <= 4.0

def test_max_len():
    c = make_canvas(6)
    place(c, piano_note(60, SR, 5.0, decay_db_s=-1), 0.5, SR)
    segs, _ = detect_segments(c, SR, DetectParams(max_len_s=2.0, end_level_db=-90.0))
    assert segs[0].end_reason == "max_len" and abs((segs[0].end - segs[0].onset) / SR - 2.0) < 0.02

def test_clicks_rejected():
    c = make_canvas(6)
    place(c, piano_note(60, SR, 2.0, peak=0.5), 1.0, SR)
    place(c, thump(SR, peak=0.005), 4.0, SR)                      # 40 dB pod tónem
    segs, rej = detect_segments(c, SR)
    assert len(segs) == 1 and len(rej) == 1
    assert rej[0].peak_db < segs[0].peak_db - 25

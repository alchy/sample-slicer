from sample_slicer.pitch import Pitch
from sample_slicer.tuning import fit_tuning_curve, assign_notes, TuningParams

def P(midi, conf=0.9):
    return Pitch(f0_hz=440.0 * 2 ** ((midi - 69) / 12), midi=midi, confidence=conf, n_votes=3, evidence=3)

def test_curve_linear_extrapolation_and_interpolation():
    curve = fit_tuning_curve([(60, 0.0), (72, 20.0)], halfwidth=0)
    assert abs(curve(66) - 10.0) < 1e-9
    assert abs(curve(84) - 40.0) < 1e-9 and abs(curve(48) + 20.0) < 1e-9   # lineárně dál
    steep = fit_tuning_curve([(60, 0.0), (61, 100.0)], halfwidth=0, max_slope=25.0)
    assert abs(steep(63) - 150.0) < 1e-9                                    # sklon omezen na 25 c/půltón
    assert fit_tuning_curve([(60, 7.0)])(90) == 7.0                          # jedna kotva = konstanta

def test_curve_median_smoothing():
    curve = fit_tuning_curve([(60, 0.0), (61, 100.0), (62, 0.0)], halfwidth=1)
    assert curve(61) == 0.0        # osamělá odchylka se vyhladí

def test_assign_sharp_treble_via_curve():
    # střed přesný, od A7 rostoucí odchylka: 105 +46 c, 107 +79 c, 108 +73 c
    pitches = [P(60.0), P(72.05), P(84.1), P(96.2), P(105.46), P(107.79), P(108.73)]
    asg, curve = assign_notes(pitches)
    assert [a.midi for a in asg] == [60, 72, 84, 96, 105, 107, 108]
    assert asg[0].anchor and asg[4].anchor is False and asg[5].reason == "curve"
    assert abs(asg[5].cents_et - 79) < 1

def test_real_petrof_top_octave():
    # skutečné hodnoty ze spiku: G6..C8 (91 -11c, 93 -6c, 95 +1c, 96 -18c, 98 0c, 100 +9c, 101 +23c, 103 +11c, 105 +46c, 107 +79c, 108 +73c)
    vals = [90.89, 92.94, 95.01, 95.82, 98.00, 100.09, 101.23, 103.11, 105.46, 107.79, 108.73]
    asg, _ = assign_notes([P(v) for v in vals])
    assert [a.midi for a in asg] == [91, 93, 95, 96, 98, 100, 101, 103, 105, 107, 108]

def test_reject_out_of_tolerance_and_range():
    asg, _ = assign_notes([P(60.0), P(64.5), P(20.0), None, P(66.0, conf=0.2)])
    assert asg[1].midi is None and asg[1].reason == "out_of_tolerance"
    assert asg[2].midi is None and asg[2].reason == "out_of_range"
    assert asg[3].midi is None and asg[3].reason == "low_confidence"
    assert asg[4].midi is None and asg[4].reason == "low_confidence"

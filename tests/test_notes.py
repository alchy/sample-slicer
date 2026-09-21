import pytest
from sample_slicer.notes import note_to_midi, midi_to_name, expand_truth

def test_note_names():
    assert note_to_midi("A0") == 21 and note_to_midi("C4") == 60 and note_to_midi("C#4") == 61
    assert note_to_midi("Db4") == 61 and note_to_midi("c8") == 108
    assert midi_to_name(60) == "C4" and midi_to_name(21) == "A0" and midi_to_name(61) == "C#4"

def test_expand_chromatic_and_major():
    assert expand_truth({"start": "A0", "pattern": "chromatic"}, 4) == [21, 22, 23, 24]
    assert expand_truth({"start": "C4", "pattern": "major"}, 8) == [60, 62, 64, 65, 67, 69, 71, 72]
    assert expand_truth({"pattern": "list", "notes": ["C4", "E4"]}, 99) == [60, 64]

def test_bad_pattern():
    with pytest.raises(ValueError):
        expand_truth({"start": "C4", "pattern": "minor"}, 3)

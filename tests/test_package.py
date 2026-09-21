import sample_slicer

def test_version():
    assert sample_slicer.__version__.startswith("2.")

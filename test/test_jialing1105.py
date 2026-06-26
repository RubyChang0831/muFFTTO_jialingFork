# test/test_jialing1105.py

from muFFTTO.jialing1105 import jialing1105_add

def test_jialing1105_add_positive_numbers():
    assert jialing1105_add(2, 3) == 5

def test_jialing1105_add_negative_numbers():
    assert jialing1105_add(-1, -2) == -3

def test_jialing1105_add_mixed_numbers():
    assert jialing1105_add(10, -4) == 6
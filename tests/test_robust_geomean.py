import numpy as np

from espai.robust_geomean import geomean_odds


def test_zero():
    ps = np.array([0.0, 0.5])
    assert geomean_odds(ps) == 0


def test_one():
    ps = np.array([1.0, 0.5])
    assert geomean_odds(ps) == 1

from distimate import Distribution


def test_distribution():
    dist = Distribution()
    assert len(dist) == 0

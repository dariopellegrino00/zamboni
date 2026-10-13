import numpy as np

from zamboni.features import features


def test_features_by_hand():
    obs = np.array(
        [
            [3.0, -1.5, 2.0, -4.0, 13.0, 4.0, 50.0],
            [-30.0, 15.0, 0.0, 0.0, 30.0, -15.0, 200.0],  # opposite corners
        ]
    )
    expected = np.array(
        [
            [0.1, -0.1, 1.0, 0.55, 0.2, -0.4, 0.25],
            [-1.0, 1.0, 6.0, -3.0, 0.0, 0.0, 1.0],
        ]
    )
    assert np.allclose(features(obs), expected)  # a batch
    assert np.allclose(features(obs[0]), expected[0])  # a single state

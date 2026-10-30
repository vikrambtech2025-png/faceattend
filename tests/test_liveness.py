import numpy as np

from faceattend.liveness import _ear


def test_ear_open_eye():
    pts = np.array([[0, 0], [1, -2], [2, -2], [4, 0], [2, 2], [1, 2]], dtype=np.float32)
    # idxs identity for test
    idxs = list(range(6))
    val = _ear(pts, idxs)
    assert val > 0.3


def test_texture_score_range():
    from faceattend.liveness import LivenessChecker

    lc = LivenessChecker(mode="passive")
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    # flat image -> low texture
    s = lc._texture_score(img)
    assert 0.0 <= s <= 1.0

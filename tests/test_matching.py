import numpy as np

from faceattend.recognition import Gallery, GalleryEntry


def test_match_known(monkeypatch, tmp_path):
    # build a fake gallery by injecting entries
    from faceattend.config import get_settings
    from faceattend.storage import make_session_factory

    sf = make_session_factory(str(get_settings().db_path))
    g = Gallery(sf)
    # inject
    v = np.ones(512, dtype=np.float32)
    v = v / np.linalg.norm(v)
    g._entries = [GalleryEntry("p1", "Alice", v)]
    # query same
    res = g.match(v)
    assert not res.is_unknown
    assert res.name == "Alice"
    assert res.distance < 0.01


def test_match_unknown():
    from faceattend.config import get_settings
    from faceattend.storage import make_session_factory

    sf = make_session_factory(str(get_settings().db_path))
    g = Gallery(sf)
    v = np.ones(512, dtype=np.float32)
    v = v / np.linalg.norm(v)
    g._entries = []
    res = g.match(v)
    assert res.is_unknown

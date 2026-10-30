from faceattend.config import get_settings


def test_defaults():
    s = get_settings()
    assert s.det_size == 640
    assert 0 < s.match_threshold < 1


def test_invalid_threshold():
    import os

    import pytest
    from pydantic import ValidationError

    os.environ["FACEATTEND_MATCH_THRESHOLD"] = "1.5"
    from faceattend.config import reset_settings

    reset_settings()
    with pytest.raises(ValidationError):
        get_settings()
    # cleanup
    del os.environ["FACEATTEND_MATCH_THRESHOLD"]
    reset_settings()

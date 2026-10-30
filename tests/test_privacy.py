from faceattend.privacy import PrivacyService
from faceattend.storage import Person, make_session_factory


def test_erase_person():
    from faceattend.config import get_settings

    sf = make_session_factory(str(get_settings().db_path))
    with sf() as s:
        p = Person(name="Bob", consent_given=True)
        s.add(p)
        s.commit()
        pid = p.id
    ps = PrivacyService(sf)
    assert ps.erase_person(pid) is True
    with sf() as s:
        got = s.get(Person, pid)
        assert got.deleted_at is not None
        assert got.name == "erased"


def test_purge_runs():
    from faceattend.config import get_settings

    sf = make_session_factory(str(get_settings().db_path))
    ps = PrivacyService(sf)
    res = ps.purge_retention()
    assert "snapshots_removed" in res
    assert "attendance_anonymized" in res

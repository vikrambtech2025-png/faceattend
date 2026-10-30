from faceattend.storage import Person, make_session_factory


def test_create_person():
    from faceattend.config import get_settings

    sf = make_session_factory(str(get_settings().db_path))
    with sf() as s:
        p = Person(name="Alice", consent_given=True)
        s.add(p)
        s.commit()
        assert p.id is not None
        got = s.get(Person, p.id)
        assert got.name == "Alice"

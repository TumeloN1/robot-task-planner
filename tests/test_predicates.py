from rtp.tasks.predicates import PREDICATES


def test_predicate_registry_has_core_relations():
    assert {"on", "near", "left_of", "inside"}.issubset(set(PREDICATES))


def test_predicates_are_callable():
    for fn in PREDICATES.values():
        assert callable(fn)

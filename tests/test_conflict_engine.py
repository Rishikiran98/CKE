from cke.graph.assertion import Assertion
from cke.graph.conflict_engine import ConflictEngine


def test_conflict_detection_for_same_subject_relation_and_overlapping_qualifiers():
    engine = ConflictEngine()
    a = Assertion(
        subject="Redis",
        relation="supports",
        object="pubsub",
        qualifiers={"temporal": "2012"},
        trust_score=0.8,
    )
    b = Assertion(
        subject="Redis",
        relation="supports",
        object="streams",
        qualifiers={"temporal": "2012"},
        trust_score=0.7,
    )

    conflicts = engine.detect_conflicts([a, b])

    assert len(conflicts) == 1


def test_conflict_resolution_prefers_highest_trust_and_marks_secondary():
    engine = ConflictEngine()
    a = Assertion("Redis", "supports", "pubsub", trust_score=0.9)
    b = Assertion("Redis", "supports", "streams", trust_score=0.4)

    winner, loser = engine.resolve_conflict(a, b)

    assert winner.object == "pubsub"
    assert loser.secondary is True

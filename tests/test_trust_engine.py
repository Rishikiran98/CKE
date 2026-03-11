import time

from cke.graph.assertion import Assertion
from cke.graph.deduplicator import AssertionDeduplicator
from cke.graph.trust_engine import TrustEngine


def test_trust_score_computation():
    engine = TrustEngine(source_weights={"wikipedia": 1.0, "unknown": 0.5}, tau=10_000)
    now = time.time()
    assertion = Assertion(
        subject="Redis",
        relation="supports",
        object="pubsub",
        source="wikipedia",
        timestamp=now,
        evidence_count=3,
        extractor_confidence=0.9,
    )

    score = engine.compute_trust(assertion, now=now)

    assert 0.5 < score < 1.0


def test_duplicate_assertion_merging_recomputes_trust():
    engine = TrustEngine(source_weights={"wikipedia": 1.0, "unknown": 0.5}, tau=10_000)
    dedup = AssertionDeduplicator(engine)

    a1 = Assertion(
        subject="Redis",
        relation="supports",
        object="pubsub",
        qualifiers={"scope": "typical"},
        source="wikipedia",
        evidence_count=1,
        extractor_confidence=0.8,
    )
    a2 = Assertion(
        subject="Redis",
        relation="supports",
        object="pubsub",
        qualifiers={"scope": "typical"},
        source="wikipedia",
        evidence_count=2,
        extractor_confidence=0.9,
    )

    merged = dedup.deduplicate([a1, a2])

    assert len(merged) == 1
    assert merged[0].evidence_count == 3
    assert merged[0].trust_score > 0

"""Tests for trust calibration and distribution stats."""

import time

from cke.graph.assertion import Assertion, Evidence
from cke.graph.trust_engine import TrustEngine
from cke.trust.calibration import TrustCalibrator, TrustCalibrationConfig


def test_trust_calibrator_formula_and_decay() -> None:
    calibrator = TrustCalibrator(
        TrustCalibrationConfig(w_src=0.5, w_freq=0.7, w_conf=1.0, tau=100.0)
    )
    trust_0 = calibrator.calibrate(1.0, 2, 0.9)
    decayed = calibrator.apply_decay(trust_0, now=110.0, observed_ts=10.0)

    assert 0.0 < trust_0 <= 1.0
    assert 0.0 < decayed <= trust_0


def test_trust_engine_fit_and_distribution_stats() -> None:
    engine = TrustEngine(config_path=None)
    now = time.time()
    assertions = [
        Assertion(
            "Redis",
            "supports",
            "PubSub",
            source="wikipedia",
            evidence=[Evidence(text="Redis supports PubSub")],
            extractor_confidence=0.9,
            evidence_count=2,
            timestamp=now,
        ),
        Assertion(
            "Redis",
            "supports",
            "Streams",
            source="unknown",
            evidence=[Evidence(text="Redis supports Streams")],
            extractor_confidence=0.5,
            evidence_count=1,
            timestamp=now,
        ),
    ]

    for assertion in assertions:
        engine.compute_trust(assertion, now=now)

    weights = engine.fit_from_graph(assertions)
    stats = engine.trust_distribution_stats(assertions)

    assert {"w_src", "w_freq", "w_conf", "tau"}.issubset(weights.keys())
    assert 0.0 <= stats["mean_trust"] <= 1.0
    assert stats["variance"] >= 0.0
    assert 0.0 <= stats["low_trust_ratio"] <= 1.0

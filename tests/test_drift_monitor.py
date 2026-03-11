import pytest
from cke.graph.drift_monitor import DriftMonitor
from cke.graph.snapshot_manager import GraphSnapshot


def test_drift_calculation_jaccard_distance():
    previous = GraphSnapshot(
        timestamp="t-1",
        assertion_hash_set=frozenset({"a", "b", "c"}),
        assertion_count=3,
    )
    current = GraphSnapshot(
        timestamp="t",
        assertion_hash_set=frozenset({"b", "c", "d"}),
        assertion_count=3,
    )

    drift = DriftMonitor.compute_drift(current, previous)

    # 1 - |{b,c}| / |{a,b,c,d}| = 1 - 2/4 = 0.5
    assert drift == 0.5


def test_smoothed_drift_uses_exponential_smoothing():
    monitor = DriftMonitor(alpha=0.2)

    smoothed = monitor.smooth_drift(delta=0.5, previous_delta=0.1)

    assert smoothed == pytest.approx(0.18)

from cke.graph.convergence_engine import ConvergenceEngine


def test_domain_becomes_stable_after_window_under_epsilon():
    engine = ConvergenceEngine(epsilon=0.05, window_size=3)

    assert engine.update("databases", 0.04) == "evolving"
    assert engine.update("databases", 0.03) == "evolving"
    assert engine.update("databases", 0.01) == "stable"


def test_domain_classified_volatile_on_high_drift():
    engine = ConvergenceEngine(epsilon=0.05, window_size=5)

    state = engine.update("cloud", 0.4)

    assert state == "volatile"

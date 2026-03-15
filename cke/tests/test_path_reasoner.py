from cke.models import Statement
from cke.reasoning.path_reasoner import PathReasoner


def test_path_reasoner_multihop_nationality_inference():
    reasoner = PathReasoner()
    context = [
        Statement("A", "directed", "B", confidence=0.9),
        Statement("B", "starred", "C", confidence=0.8),
        Statement("C", "nationality", "USA", confidence=0.7),
    ]

    answer = reasoner.answer("What is A nationality?", context)

    assert answer == "USA"
    trace = reasoner.format_reasoning_path(context)
    assert "A -> directed -> B" in trace
    assert "B -> starred -> C" in trace
    assert "C -> nationality -> USA" in trace
    assert "Path confidence = 0.5040" in trace


def test_path_reasoner_applies_located_in_transitivity_rule():
    reasoner = PathReasoner()
    context = [
        Statement("District", "located_in", "City", confidence=0.9),
        Statement("City", "located_in", "Country", confidence=0.8),
    ]

    answer = reasoner.answer("Where is District located in?", context)

    assert answer == "Country"
    trace = reasoner.format_reasoning_path(context)
    assert "Rule applied located_in_transitivity" in trace

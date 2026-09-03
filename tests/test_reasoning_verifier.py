from cke.models import Statement
from cke.reasoning.pattern_memory import PatternMemory
from cke.reasoning.verifier import ReasoningVerifier


def test_verifier_fails_when_required_evidence_missing():
    verifier = ReasoningVerifier(confidence_threshold=0.7)
    context = [Statement("A", "nationality", "US", confidence=0.9)]

    result = verifier.verify(
        query="Do A and B have same nationality?",
        context=context,
        reasoning_path=context,
        answer="yes",
        confidence_score=0.95,
        required_facts=[("A", "nationality"), ("B", "nationality")],
        operator_checks=[
            {"operator": "equality", "inputs": ("US", "US"), "result": True}
        ],
    )

    assert not result.passed
    assert not result.evidence_complete


def test_pattern_memory_executes_nationality_template():
    memory = PatternMemory()
    context = [
        Statement("Alice", "nationality", "US", confidence=0.8),
        Statement("Bob", "nationality", "US", confidence=0.9),
    ]

    execution = memory.execute("Do Alice and Bob have same nationality?", context)

    assert execution is not None
    assert execution.answer == "yes"
    assert "equality" in execution.operators_used

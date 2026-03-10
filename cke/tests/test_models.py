from cke.models import Statement


def test_statement_backward_compatibility_defaults():
    st = Statement("Redis", "supports", "PubSub")
    assert st.subject == "Redis"
    assert st.context == {}
    assert st.confidence == 1.0
    assert st.source is None
    assert st.timestamp is None
    assert st.as_text() == "Redis supports PubSub"


def test_statement_supports_contextual_fields():
    st = Statement(
        "Redis",
        "uses",
        "RESP",
        context={"section": "overview"},
        confidence=0.93,
        source="doc://redis",
        timestamp="2026-03-10T00:00:00Z",
    )
    assert st.context["section"] == "overview"
    assert st.confidence == 0.93
    assert st.source == "doc://redis"
    assert st.timestamp == "2026-03-10T00:00:00Z"

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


def test_statement_supports_pipeline_metadata_fields():
    st = Statement(
        "Albert Einstein",
        "nationality",
        "German",
        statement_id="st-1",
        chunk_id="chunk-1",
        source_doc_id="doc-1",
        canonical_subject_id="ent-albert-einstein",
        canonical_object_id="ent-german",
        trust_score=0.88,
        retrieval_score=0.91,
        supporting_span=(12, 37),
    )

    assert st.statement_id == "st-1"
    assert st.chunk_id == "chunk-1"
    assert st.source_doc_id == "doc-1"
    assert st.canonical_subject_id == "ent-albert-einstein"
    assert st.canonical_object_id == "ent-german"
    assert st.trust_score == 0.88
    assert st.retrieval_score == 0.91
    assert st.supporting_span == (12, 37)

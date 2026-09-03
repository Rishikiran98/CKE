import json

from cke.extractor.llm_extractor import (
    LLMConfig,
    LLMExtractor,
    validate_qualifiers,
)
from cke.extractor.rule_extractor import RuleExtractor
from cke.graph.assertion_validator import AssertionValidator


def test_rule_extractor_detects_expected_relations():
    extractor = RuleExtractor()
    text = "Redis is a database. Redis uses RESP. Redis located in memory."
    statements = extractor.extract(text)
    triples = {(s.subject, s.relation, s.object) for s in statements}
    assert ("Redis", "uses", "RESP") in triples
    assert ("Redis", "located_in", "memory") in triples


def test_llm_extractor_parses_valid_json_payload():
    extractor = LLMExtractor(config=LLMConfig(api_key="dummy"))
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        "["
                        '{"subject":"Redis","relation":"uses",'
                        '"object":"RESP","confidence":0.9}'
                        "]"
                    )
                }
            }
        ]
    }
    statements = extractor._parse_response(payload)
    assert len(statements) == 1
    assert statements[0].subject == "Redis"
    assert statements[0].relation == "uses"


def test_rule_extractor_reduces_a_rambling_object_to_its_head_noun():
    """A trailing clause must not become an entity.

    This previously discarded every is_a statement to avoid objects like
    "song from the 1968 musical film featurette that keeps going". Dropping a
    whole relation type was the wrong instrument: the object is now cut at its
    modifier boundary, so the relation is kept and the garbage is not.
    """
    extractor = RuleExtractor()
    text = (
        "Ed Wood is a song from the 1968 musical film featurette that keeps going. "
        "Scott Derrickson uses practical effects in a very long production workflow "
        "that includes many extra details and descriptors."
    )

    statements = extractor.extract(text)
    triples = {(s.subject, s.relation, s.object) for s in statements}

    # The is_a statement is kept, and its object is the head noun.
    assert ("Ed Wood", "is_a", "song") in triples

    # The same cut applies to every relation, not just is_a.
    assert ("Scott Derrickson", "uses", "practical effects") in triples

    assert all(len(obj) <= extractor.MAX_OBJECT_LENGTH for _, _, obj in triples)
    # No object may still carry the clause it was cut from.
    assert not any(" that " in obj or " which " in obj for _, _, obj in triples)


def test_rule_extractor_keeps_the_subject_clear_of_the_copula():
    """ "Redis is located in memory" yielded the subject "Redis is".

    That made "Redis" and "Redis is" separate entities, so a path through the
    two statements about Redis could not connect.
    """
    extractor = RuleExtractor()

    statements = extractor.extract(
        "Redis is located in memory. Redis uses RESP protocol."
    )
    subjects = {s.subject for s in statements}

    assert subjects == {"Redis"}


def test_rule_extractor_strips_a_leading_determiner_from_an_object():
    """ "a text protocol" and "text protocol" must not be separate nodes."""
    extractor = RuleExtractor()

    statements = extractor.extract(
        "Memcached uses a text protocol. Redis uses the text protocol."
    )
    objects = {s.object for s in statements}

    assert objects == {"text protocol"}


def test_rule_extractor_rejects_an_object_with_no_entity_left():
    """Cutting can leave nothing; that is not a statement."""
    extractor = RuleExtractor()

    assert extractor.extract("Redis uses it.") == []
    assert extractor._is_valid_statement("Redis", "is_a", "the") is False
    assert extractor._is_valid_statement("Redis", "is_a", "Redis") is False


def test_rule_extractor_truncates_on_a_word_boundary():
    """Cutting mid-word invents an entity that matches nothing."""
    extractor = RuleExtractor()
    long_object = " ".join(["alpha"] * 40)

    statements = extractor.extract(f"Thing uses {long_object}")

    assert statements, "expected an extraction"
    obj = statements[0].object
    assert len(obj) <= extractor.MAX_OBJECT_LENGTH
    assert not obj.endswith("alph"), "the final word was split"
    assert all(word == "alpha" for word in obj.split())


def test_llm_extractor_captures_temporal_and_conditional_qualifiers():
    """Extraction preserves temporal and conditional qualifiers from LLM output."""
    text = "Einstein lived in Germany from 1879 to 1933 when he emigrated."
    extractor = LLMExtractor(config=LLMConfig(api_key="dummy"))

    span_text = "Einstein lived in Germany from 1879 to 1933"
    start = text.index(span_text)
    end = start + len(span_text)

    assertions = [
        {
            "subject": "Einstein",
            "relation": "lived_in",
            "object": {"type": "entity", "value": "Germany"},
            "qualifiers": {
                "temporal": {"start": "1879", "end": "1933"},
                "condition": {"when": "before emigration"},
            },
            "extractor_confidence": 0.92,
            "confidence": 0.95,
            "evidence": [
                {
                    "chunk_id": "chunk-0",
                    "span_start": start,
                    "span_end": end,
                    "text": span_text,
                    "extractor_confidence": 0.92,
                    "source_weight": 1.0,
                }
            ],
        }
    ]
    payload = {"choices": [{"message": {"content": json.dumps(assertions)}}]}
    statements = extractor._parse_response(payload, source_text=text)

    assert len(statements) == 1
    st = statements[0]
    assert st.subject == "Einstein"
    assert st.object == "Germany"
    # Temporal qualifiers preserved
    assert st.qualifiers["temporal"]["start"] == "1879"
    assert st.qualifiers["temporal"]["end"] == "1933"
    # Conditional qualifier preserved
    assert st.qualifiers["condition"]["when"] == "before emigration"
    # Extractor confidence stored in context
    assert st.context["extractor_confidence"] == 0.92
    # Passes validator
    assert AssertionValidator().validate(st)


def test_llm_extractor_rejects_invalid_qualifier_keys():
    """Assertions with non-schema qualifier keys are rejected."""
    extractor = LLMExtractor(config=LLMConfig(api_key="dummy"))
    assertions = [
        {
            "subject": "Redis",
            "relation": "uses",
            "object": "RESP",
            "confidence": 0.9,
            "qualifiers": {"bogus_key": "value"},
        }
    ]
    payload = {"choices": [{"message": {"content": json.dumps(assertions)}}]}
    statements = extractor._parse_response(payload)
    assert len(statements) == 0


def test_llm_extractor_handles_typed_object_dict():
    """Object field as {type, value} dict is normalized to plain string."""
    extractor = LLMExtractor(config=LLMConfig(api_key="dummy"))
    assertions = [
        {
            "subject": "Python",
            "relation": "version",
            "object": {"type": "literal", "value": "3.12"},
            "confidence": 0.99,
            "qualifiers": {"scope": {"version": "3.12"}},
            "extractor_confidence": 0.97,
        }
    ]
    payload = {"choices": [{"message": {"content": json.dumps(assertions)}}]}
    statements = extractor._parse_response(payload)
    assert len(statements) == 1
    assert statements[0].object == "3.12"
    assert statements[0].qualifiers["scope"]["version"] == "3.12"
    assert statements[0].context["extractor_confidence"] == 0.97


def test_validate_qualifiers_accepts_valid_structures():
    assert validate_qualifiers({}) is True
    assert validate_qualifiers({"modality": "typical"}) is True
    assert validate_qualifiers({"temporal": {"start": "2020"}}) is True
    assert validate_qualifiers({"condition": {"environment": "prod"}}) is True
    assert validate_qualifiers({"scope": {"jurisdiction": "US"}}) is True


def test_validate_qualifiers_rejects_invalid_structures():
    assert validate_qualifiers({"unknown_key": "x"}) is False
    assert validate_qualifiers({"modality": "invalid_value"}) is False
    assert validate_qualifiers({"temporal": {"bad_key": "v"}}) is False
    assert validate_qualifiers({"condition": "not_a_dict"}) is False

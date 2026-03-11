from cke.extractor.llm_extractor import LLMConfig, LLMExtractor
from cke.extractor.rule_extractor import RuleExtractor


def test_rule_extractor_detects_expected_relations():
    extractor = RuleExtractor()
    text = "Redis is a database. Redis uses RESP. Redis located in memory."
    statements = extractor.extract(text)
    triples = {(s.subject, s.relation, s.object) for s in statements}
    assert ("Redis", "is_a", "database") in triples
    assert ("Redis", "uses", "RESP") in triples
    assert ("Redis", "located_in", "memory") in triples


def test_llm_extractor_parses_valid_json_payload():
    extractor = LLMExtractor(config=LLMConfig(api_key="dummy"))
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '[{"subject":"Redis","relation":"uses","object":"RESP","confidence":0.9}]'
                    )
                }
            }
        ]
    }
    statements = extractor._parse_response(payload)
    assert len(statements) == 1
    assert statements[0].subject == "Redis"
    assert statements[0].relation == "uses"

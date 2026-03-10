from cke.extractor.llm_extractor import LLMConfig, LLMExtractor


def test_llm_extractor_falls_back_to_rule_extractor_without_api_key():
    extractor = LLMExtractor(config=LLMConfig(api_key=None))
    statements = extractor.extract("Redis supports PubSub messaging.")
    triples = {(s.subject, s.relation, s.object) for s in statements}
    assert ("Redis", "supports", "PubSub") in triples


def test_llm_extractor_parses_openai_style_response():
    extractor = LLMExtractor(config=LLMConfig(api_key="dummy"))
    payload = {
        "choices": [
            {
                "message": {
                    "content": '{"triples": [{"subject": "Redis", "relation": "implemented via", "object": "RESP"}]}'
                }
            }
        ]
    }
    statements = extractor._parse_response(payload)
    assert len(statements) == 1
    assert statements[0].relation == "implemented_via"

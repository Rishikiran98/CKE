from cke.models import Statement
from cke.reasoning.llm_reasoner import LLMReasoner, LLMReasonerConfig
from cke.reasoning.reasoner import TemplateReasoner


def test_llm_reasoner_falls_back_without_api_key():
    context = [Statement("Redis", "uses", "RESP")]
    llm_reasoner = LLMReasoner(config=LLMReasonerConfig(api_key=None))
    template_answer = TemplateReasoner().answer(
        "What protocol does Redis use?", context
    )
    assert (
        llm_reasoner.answer("What protocol does Redis use?", context)
        == template_answer
    )


def test_llm_reasoner_parses_openai_style_response_payload():
    reasoner = LLMReasoner(config=LLMReasonerConfig(api_key="dummy"))
    payload = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"answer": "Redis uses RESP protocol.", '
                        '"used_evidence": ["Redis uses RESP"]}'
                    )
                }
            }
        ]
    }
    assert reasoner._parse_answer(payload) == "Redis uses RESP protocol."


def test_template_reasoner_behavior_unchanged():
    reasoner = TemplateReasoner()
    context = [Statement("Redis", "uses", "RESP")]
    assert (
        reasoner.answer("What protocol does Redis use?", context)
        == "Redis uses RESP protocol."
    )

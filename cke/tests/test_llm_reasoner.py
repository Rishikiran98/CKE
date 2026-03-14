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
        llm_reasoner.answer("What protocol does Redis use?", context) == template_answer
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
    assert reasoner._parse_answer(payload) == "Redis uses RESP protocol"


def test_template_reasoner_behavior_unchanged():
    reasoner = TemplateReasoner()
    context = [Statement("Redis", "uses", "RESP")]
    assert (
        reasoner.answer("What protocol does Redis use?", context)
        == "Redis uses RESP protocol."
    )


def test_llm_reasoner_normalizes_dict_evidence_for_fallback():
    llm_reasoner = LLMReasoner(config=LLMReasonerConfig(api_key=None))
    context = [
        {
            "subject": "Redis",
            "relation": "uses",
            "object": "RESP",
            "trust_score": 0.91,
        }
    ]
    assert (
        llm_reasoner.answer("What protocol does Redis use?", context)
        == "Redis uses RESP protocol."
    )


def test_llm_reasoner_prompt_is_question_anchored():
    llm_reasoner = LLMReasoner(config=LLMReasonerConfig(api_key="dummy"))
    prompt = llm_reasoner._build_prompt(
        "What protocol does Redis use?",
        [Statement("Redis", "uses", "RESP", confidence=0.91)],
    )

    assert "Task: answer the QUESTION" in prompt
    assert "QUESTION: What protocol does Redis use?" in prompt
    assert "[E1] Redis uses RESP" in prompt


def test_llm_reasoner_limits_and_ranks_context_for_question_relevance():
    reasoner = LLMReasoner(config=LLMReasonerConfig(api_key="dummy"))
    context = [
        Statement(f"Noise{i}", "related_to", f"Thing{i}", confidence=0.99)
        for i in range(12)
    ]
    context.append(Statement("Animorphs", "is", "Book series", confidence=0.6))

    selected = reasoner._select_context("What is Animorphs?", context, limit=10)

    assert len(selected) == 10
    assert any(st.subject == "Animorphs" for st in selected)


def test_llm_reasoner_normalizes_short_span_answers():
    reasoner = LLMReasoner(config=LLMReasonerConfig(api_key="dummy"))

    assert reasoner._normalize_answer("The answer is Animorphs.") == "Animorphs"
    assert reasoner._normalize_answer('"Chief of Protocol"') == "Chief of Protocol"
    assert reasoner._normalize_answer("yes") == "yes"

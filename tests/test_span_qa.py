"""Tests for the span answerer.

The defect these guard against is structural, not one of accuracy: the
previous answerer returned the first fifty words of a sentence, so exact match
against a short gold answer was zero whatever retrieval did. The properties
below are the ones that make a metric capable of moving, plus the type
agreement the scoring table documents.

The prose in these tests is a unit fixture, not evaluation data. Nothing here
produces a reported metric; the numbers in the pull request come from real
HotpotQA dev items.
"""

from __future__ import annotations

import pytest

from cke.evaluation.span_qa import (
    _MAX_SPAN_TOKENS,
    AnswerType,
    SpanExtractiveQA,
)

_WORDS = __import__("re").compile(r"\w+")

CONTEXT = (
    "SpaceX was founded in 2002 by Elon Musk. "
    "The company is based in Hawthorne, California. "
    "It has launched 250 missions since then."
)


@pytest.fixture()
def qa() -> SpanExtractiveQA:
    return SpanExtractiveQA()


def test_an_answer_is_short_enough_to_match_a_gold_string(qa):
    """The guarantee the old answerer broke.

    A fifty-word sentence prefix cannot equal "Elon Musk" under any
    normalisation, so exact match was zero by construction.
    """
    long_context = CONTEXT + " " + ("Filler prose about rockets. " * 200)
    answer = qa.answer("Who founded SpaceX?", long_context)

    assert 0 < len(_WORDS.findall(answer)) <= _MAX_SPAN_TOKENS


def test_the_answer_is_drawn_from_the_context(qa):
    answer = qa.answer("Who founded SpaceX?", CONTEXT)
    assert answer in CONTEXT


def test_the_same_question_and_context_give_the_same_answer(qa):
    """Symmetry across comparison arms.

    Both arms of the benchmark answer through this class. It holds no state,
    so an arm cannot be answered differently from another by construction
    rather than by convention.
    """
    first = SpanExtractiveQA().answer("Who founded SpaceX?", CONTEXT)
    second = SpanExtractiveQA().answer("Who founded SpaceX?", CONTEXT)
    repeated = [qa.answer("Who founded SpaceX?", CONTEXT) for _ in range(3)]

    assert first == second
    assert set(repeated) == {first}


def test_empty_input_gives_an_empty_answer(qa):
    assert qa.answer("Who founded SpaceX?", "") == ""
    assert qa.answer("Who founded SpaceX?", "   ") == ""
    assert qa.answer("", CONTEXT) == ""


def test_a_date_question_is_not_answered_with_a_name(qa):
    assert qa.answer("In what year was SpaceX founded?", CONTEXT) == "2002"


def test_a_person_question_is_not_answered_with_a_year(qa):
    assert qa.answer("Who founded SpaceX?", CONTEXT) == "Elon Musk"


def test_a_span_that_only_repeats_the_question_is_not_returned(qa):
    """A candidate whose every token is already in the question is not news."""
    answer = qa.answer("Who founded SpaceX?", CONTEXT)
    assert answer != "SpaceX"


def test_a_one_character_span_taken_from_the_question_is_penalised(qa):
    """The repetition penalty is tested against every question token.

    Content terms drop one-character tokens, so a span such as "S" — lifted
    straight out of a question about the album "2014 S/S" — used to escape the
    penalty written for exactly that case.
    """
    context = "2014 S/S is the debut album of the South Korean group Winner."
    answer = qa.answer("2014 S/S is the debut album of which group?", context)

    assert answer != "S"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("How many missions has it launched?", AnswerType.NUMBER),
        ("How much did it cost?", AnswerType.NUMBER),
        ("How did it happen?", AnswerType.ENTITY),
        ("In what year was it founded?", AnswerType.DATE),
        ("When was it founded?", AnswerType.DATE),
        ("Who founded it?", AnswerType.PERSON),
        ("Where is it based?", AnswerType.PLACE),
        ("Is it based in California?", AnswerType.BOOLEAN),
        ("What does it build?", AnswerType.ENTITY),
    ],
)
def test_question_type_classification(question, expected):
    """ "how many" must be tested before the bare "how"."""
    assert SpanExtractiveQA.classify_question(question) == expected


def test_a_yes_no_question_is_answered_yes_or_no(qa):
    """Crude by design, and the test says no more than the code promises.

    The heuristic has no notion of negation, so this asserts the shape of the
    answer rather than its correctness.
    """
    assert qa.answer("Is SpaceX based in Hawthorne?", CONTEXT) in {"yes", "no"}
    assert qa.answer("Was Rome built by penguins?", CONTEXT) in {"yes", "no"}


def test_a_capitalised_run_longer_than_the_cap_is_not_returned(qa):
    """The cap is what keeps an answer answer-shaped.

    A long title-case run is a single proper-noun match, and without the cap
    it is a candidate like any other.
    """
    context = (
        "The Lord High Chancellor Of The Exchequer And Keeper Of The Great "
        "Seal opened the session."
    )
    answer = qa.answer("Who opened the session?", context)

    assert len(_WORDS.findall(answer)) <= _MAX_SPAN_TOKENS


def test_a_context_with_no_content_words_gives_no_answer(qa):
    """A span of only stop words is not an answer, however it scores.

    "It" heads many sentences and matches the proper-noun pattern.
    """
    assert qa.answer("Who founded SpaceX?", "It is that.") == ""


def test_the_nearer_of_two_names_answers_a_place_question(qa):
    """Proximity is the only thing separating these two candidates.

    Without a named-entity model a person and a place are both capitalised
    spans, so type agreement scores them identically. What is left is that an
    answer sits near the question terms its sentence shares.
    """
    assert qa.answer("Where is SpaceX based?", CONTEXT) == "Hawthorne"

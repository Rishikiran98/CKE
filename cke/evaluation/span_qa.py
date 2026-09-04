"""Extract a short answer span from retrieved context.

This replaces an answerer that returned the first fifty words of the
highest-overlap sentence. That could never produce an exact match: gold
answers are short strings like "Elon Musk", and a sentence prefix normalises
to dozens of tokens. Exact match was therefore zero by construction, whatever
retrieval did, and F1 measured how much of a sentence happened to overlap a
two-word answer.

What this is
------------
A lexical span baseline with no learned components. It infers the expected
answer type from the question's interrogative, generates typed candidate spans
from the context, and scores them by type agreement, question overlap and
proximity. This is the shape of pre-neural IR question answering, and it is
weak by modern standards.

What it is not is an approximation of a reader model. It cannot resolve
coreference, cannot compose across sentences, has no notion of negation, and
will miss any answer not present as a contiguous surface string. Where a
language model is available it should be used instead; this exists so that a
run without one still produces a metric capable of moving.

Symmetry
--------
The algorithm sees only a question and a context string. It cannot tell which
retrieval strategy produced the context, so every arm of a comparison is
answered identically by construction rather than by convention.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["AnswerType", "SpanKind", "SpanExtractiveQA"]


class AnswerType:
    """The kind of thing a question asks for."""

    PERSON = "person"
    DATE = "date"
    PLACE = "place"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ENTITY = "entity"


class SpanKind:
    """The kind of thing a candidate span is, by surface form alone.

    These are surface classes, not entity classes. Without a named-entity
    model a capitalised span cannot be told from a place, an organisation or a
    title, so all of them are PROPER.
    """

    DATE = "date"
    NUMBER = "number"
    PROPER = "proper"
    CHUNK = "chunk"


#: A proper-noun sequence: capitalised words, optionally joined by the lowercase
#: particles that occur inside real names ("Bank of America", "Ludwig van
#: Beethoven").
_PROPER_NOUN = re.compile(
    r"\b[A-Z][\w.'’-]*"
    r"(?:\s+(?:of|de|van|der|von|del|la|le|the)\s+[A-Z][\w.'’-]*"
    r"|\s+[A-Z][\w.'’-]*)*"
)

#: A four-digit year, the date form that dominates in encyclopedic text.
_YEAR = re.compile(r"\b(?:1[0-9]{3}|20[0-9]{2})\b")

#: A fuller date, so "March 3, 1847" is preferred over the bare year inside it.
_FULL_DATE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October"
    r"|November|December)\s+\d{1,2},?\s+\d{4}\b"
)

_NUMBER = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")

#: A run of lowercase words, the fallback candidate source for answers that
#: are not names, dates or numbers ("a binary protocol").
_LOWERCASE_RUN = re.compile(r"\b[a-z][\w-]*(?:\s+[a-z][\w-]*){0,2}\b")

_WORD = re.compile(r"\w+")

#: Words that carry no discriminating power when matching a question to a span.
_STOP_WORDS = frozenset(
    """a an the of in on at to for from by with and or is are was were be been
    being do does did what which who whom whose when where why how that this
    these those it its his her their there than then as also""".split()
)

#: Interrogative to expected answer type. Ordered: the first match wins, so
#: "how many" is tested before the bare "how".
_QUESTION_TYPES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bhow\s+many\b|\bhow\s+much\b", re.I), AnswerType.NUMBER),
    (
        re.compile(r"\bwhat\s+year\b|\bwhich\s+year\b|\bin\s+what\s+year\b", re.I),
        AnswerType.DATE,
    ),
    (re.compile(r"^\s*when\b|\bwhat\s+date\b", re.I), AnswerType.DATE),
    (re.compile(r"^\s*who\b|^\s*whom\b|^\s*whose\b", re.I), AnswerType.PERSON),
    (
        re.compile(r"^\s*where\b|\bwhat\s+(?:country|city|state|place)\b", re.I),
        AnswerType.PLACE,
    ),
    (
        re.compile(
            r"^\s*(?:are|is|was|were|do|does|did|can|could|has|have|had)\b", re.I
        ),
        AnswerType.BOOLEAN,
    ),
)

#: How well each surface class answers each question type. A "who" question
#: answered with a year is wrong even when the year sits in the best sentence;
#: a "what" question can legitimately be answered by any of them.
#:
#: PERSON and PLACE share a row because this baseline cannot distinguish a
#: person from a place without a named-entity model. Both are names, so both
#: prefer a capitalised span and reject a bare number.
_TYPE_AGREEMENT: dict[str, dict[str, float]] = {
    AnswerType.PERSON: {
        SpanKind.PROPER: 3.0,
        SpanKind.CHUNK: -2.0,
        SpanKind.DATE: -3.0,
        SpanKind.NUMBER: -3.0,
    },
    AnswerType.PLACE: {
        SpanKind.PROPER: 3.0,
        SpanKind.CHUNK: -2.0,
        SpanKind.DATE: -3.0,
        SpanKind.NUMBER: -3.0,
    },
    AnswerType.DATE: {
        SpanKind.DATE: 3.0,
        SpanKind.NUMBER: -1.0,
        SpanKind.PROPER: -3.0,
        SpanKind.CHUNK: -3.0,
    },
    AnswerType.NUMBER: {
        SpanKind.NUMBER: 3.0,
        SpanKind.DATE: 0.0,
        SpanKind.PROPER: -3.0,
        SpanKind.CHUNK: -3.0,
    },
    AnswerType.ENTITY: {
        SpanKind.PROPER: 2.0,
        SpanKind.CHUNK: 1.0,
        SpanKind.DATE: 0.0,
        SpanKind.NUMBER: 0.0,
    },
}

#: Longest span considered, in tokens. Gold answers are short; allowing long
#: spans would reintroduce the sentence-prefix problem in another form.
_MAX_SPAN_TOKENS = 6


@dataclass(frozen=True)
class _Candidate:
    text: str
    kind: str
    sentence_index: int
    char_start: int


class SpanExtractiveQA:
    """Answer a question with a short span drawn from the context."""

    #: How the answers were produced, for printing beside a figure. The LLM
    #: answerer exposes the same property; the benchmark prints whichever ran.
    description = "SpanExtractiveQA — lexical span baseline, no language model"

    #: No model reads the context here, so a figure produced with this
    #: answerer is the baseline's own. A summary carries this so that the
    #: distinction survives into the results file.
    uses_language_model = False

    def answer(self, question: str, context: str) -> str:
        if not context.strip() or not question.strip():
            return ""

        expected = self.classify_question(question)
        if expected == AnswerType.BOOLEAN:
            return self._answer_boolean(question, context)

        sentences = self._split_sentences(context)
        if not sentences:
            return ""

        question_terms = self._content_terms(question)
        question_tokens = {token.lower() for token in _WORD.findall(question)}
        sentence_scores = [
            self._sentence_score(sentence, question_terms) for sentence in sentences
        ]

        best, best_score = "", float("-inf")
        for candidate in self._candidates(sentences):
            score = self._score(
                candidate,
                expected=expected,
                question_terms=question_terms,
                question_tokens=question_tokens,
                sentence=sentences[candidate.sentence_index],
                sentence_score=sentence_scores[candidate.sentence_index],
            )
            if score > best_score:
                best, best_score = candidate.text, score
        return best

    # ------------------------------------------------------------------
    # Question analysis
    # ------------------------------------------------------------------

    @staticmethod
    def classify_question(question: str) -> str:
        """Infer the expected answer type from the interrogative."""
        for pattern, answer_type in _QUESTION_TYPES:
            if pattern.search(question):
                return answer_type
        return AnswerType.ENTITY

    @staticmethod
    def _content_terms(text: str) -> set[str]:
        return {
            token
            for token in _WORD.findall(text.lower())
            if token not in _STOP_WORDS and len(token) > 1
        }

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(context: str) -> list[str]:
        return [s.strip() for s in re.split(r"[.!?\n]+", context) if s.strip()]

    def _candidates(self, sentences: list[str]) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for index, sentence in enumerate(sentences):
            for pattern, kind in (
                (_FULL_DATE, SpanKind.DATE),
                (_YEAR, SpanKind.DATE),
                (_NUMBER, SpanKind.NUMBER),
                (_PROPER_NOUN, SpanKind.PROPER),
                (_LOWERCASE_RUN, SpanKind.CHUNK),
            ):
                for match in pattern.finditer(sentence):
                    candidate = self._candidate(match, kind, index)
                    if candidate is not None:
                        candidates.append(candidate)
        return candidates

    @staticmethod
    def _candidate(
        match: re.Match[str], kind: str, sentence_index: int
    ) -> _Candidate | None:
        """Trim a match into a candidate, or reject it.

        A span that is only stop words is not an answer, however well it sits
        in the sentence: "The" heads many sentences and matches the
        proper-noun pattern.
        """
        text = match.group(0).strip(" ,;:'\"")
        tokens = _WORD.findall(text.lower())
        if not tokens or len(tokens) > _MAX_SPAN_TOKENS:
            return None
        if all(token in _STOP_WORDS for token in tokens):
            return None
        return _Candidate(text, kind, sentence_index, match.start())

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _sentence_score(sentence: str, question_terms: set[str]) -> float:
        tokens = set(_WORD.findall(sentence.lower()))
        if not tokens or not question_terms:
            return 0.0
        return len(question_terms & tokens) / len(question_terms)

    def _score(
        self,
        candidate: _Candidate,
        *,
        expected: str,
        question_terms: set[str],
        question_tokens: set[str],
        sentence: str,
        sentence_score: float,
    ) -> float:
        tokens = {token.lower() for token in _WORD.findall(candidate.text)}

        # The sentence the span sits in has to be about the question.
        score = 2.0 * sentence_score

        score += _TYPE_AGREEMENT[expected][candidate.kind]

        # A span that merely repeats the question is not an answer. Tested
        # against every token of the question, not against its content terms:
        # content terms drop one-character tokens, so a span such as "S", taken
        # from a question about the album "2014 S/S", escaped the penalty that
        # exists for exactly that case.
        if tokens <= question_tokens:
            score -= 4.0
        else:
            score -= 1.5 * (len(tokens & question_terms) / len(tokens))

        # An answer usually sits near the terms its sentence shares with the
        # question.
        score += self._proximity(candidate, sentence, question_terms)

        # A full date beats the bare year inside it.
        if candidate.kind == SpanKind.DATE and len(tokens) > 1:
            score += 0.5
        return score

    @staticmethod
    def _proximity(
        candidate: _Candidate, sentence: str, question_terms: set[str]
    ) -> float:
        positions = [
            match.start()
            for match in _WORD.finditer(sentence.lower())
            if match.group(0) in question_terms
        ]
        if not positions:
            return 0.0
        distance = min(abs(candidate.char_start - pos) for pos in positions)
        return 1.0 / (1.0 + distance / 40.0)

    # ------------------------------------------------------------------
    # Yes/no questions
    # ------------------------------------------------------------------

    @staticmethod
    def _answer_boolean(question: str, context: str) -> str:
        """Answer a yes/no question by whether the context supports it.

        Deliberately crude: it reports "yes" when the question's content terms
        are well covered by the context and "no" otherwise. It carries no
        notion of negation, so a context that explicitly contradicts the
        question still reads as support, and it cannot compare two entities,
        which is what most yes/no multi-hop questions actually ask.
        """
        terms = SpanExtractiveQA._content_terms(question)
        if not terms:
            return "no"
        context_tokens = set(_WORD.findall(context.lower()))
        coverage = len(terms & context_tokens) / len(terms)
        return "yes" if coverage >= 0.6 else "no"

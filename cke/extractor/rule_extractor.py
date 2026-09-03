"""Regex-based extraction of simple relations.

This extractor recognises four sentence forms. It is deliberately shallow: it
is a baseline, not a parser, and everything it cannot express is left
unextracted rather than approximated.

Two properties matter for anything measured downstream. An object must be the
head noun phrase, not the rest of the sentence, or the graph fills with nodes
like "song from the 1968 musical film featurette that keeps going" that nothing
can ever match against. And a subject must not absorb the copula that precedes
the relation, or "Redis" and "Redis is" become separate entities and a path
through them breaks.
"""

from __future__ import annotations

import re

from cke.models import Statement

#: Determiners stripped from the front of an object. "a text protocol" and
#: "text protocol" are the same entity, and keeping both fragments the graph.
_LEADING_DETERMINERS = re.compile(r"^(?:a|an|the)\s+", re.I)

#: A parenthetical is an aside, never part of the name a later sentence uses.
#: Encyclopedic prose is full of them — "The Laleli Mosque (Turkish: "Laleli
#: Camii, or Tulip Mosque")" — and carrying one into the subject makes an
#: entity that nothing else in the graph can match. The closing bracket is
#: optional because the sentence splitter cuts inside asides that contain a
#: full stop.
_PARENTHETICAL = re.compile(r"\s*[(\[][^)\]]*[)\]]?\s*")

#: A subject holding a finite copula is a clause, not a name: everything from
#: the copula onwards is predicate. The subject group is non-greedy and starts
#: at the sentence, so "The Sultan Ahmed Mosque is a historic mosque located
#: in Istanbul" gives the located_in pattern the subject "The Sultan Ahmed
#: Mosque is a historic mosque". Cutting at the copula leaves the entity.
#:
#: The same cut handles a copula in final position, which is what a subject
#: ending at the relation looks like: "Redis is located in memory" yields the
#: subject "Redis is".
_COPULA = re.compile(r"\s+(?:is|was|are|were|has|have|had)(?:\s|$)", re.I)

#: Where a modifier clause begins. The object is cut here so it is the head
#: noun phrase rather than the remainder of the sentence.
#:
#: "of" is deliberately absent: it is part of entity names such as "Bank of
#: America". The cost of that choice is that a genuine modifier introduced by
#: "of" is kept. The cost of the reverse would be truncating real names, which
#: is worse for a graph keyed on them.
_MODIFIER_BOUNDARY = re.compile(
    r"\s+(?:that|which|who|whom|whose|where|when|from|in|with|for|by|during)\s+",
    re.I,
)

#: An object that reduces to one of these carries no entity.
_EMPTY_OBJECTS = frozenset(
    {"a", "an", "the", "it", "its", "this", "that", "these", "those", ""}
)


class RuleExtractor:
    """Extract simple assertion patterns from free text."""

    MAX_OBJECT_LENGTH = 80

    PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"(?P<s>[^.]+?)\s+is\s+a\s+(?P<o>[^.]+)", re.I), "is_a"),
        (re.compile(r"(?P<s>[^.]+?)\s+uses\s+(?P<o>[^.]+)", re.I), "uses"),
        (re.compile(r"(?P<s>[^.]+?)\s+developed\s+(?P<o>[^.]+)", re.I), "developed"),
        (
            re.compile(r"(?P<s>[^.]+?)\s+located\s+in\s+(?P<o>[^.]+)", re.I),
            "located_in",
        ),
    )

    def extract(self, text: str) -> list[Statement]:
        assertions: list[Statement] = []
        for sentence in self._split_sentences(text):
            for pattern, relation in self.PATTERNS:
                match = pattern.search(sentence)
                if not match:
                    continue
                subject = self._normalize_entity(match.group("s"))
                obj = self._normalize_object(match.group("o"))
                if not self._is_valid_statement(subject, relation, obj):
                    continue
                assertions.append(
                    Statement(
                        subject=subject,
                        relation=relation,
                        object=obj,
                        confidence=1.0,
                    )
                )
        return assertions

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"[.!?]\s*", text) if s.strip()]

    @staticmethod
    def _clean(token: str) -> str:
        return re.sub(r"\s+", " ", token.strip(" ,;\n\t"))

    def _normalize_entity(self, token: str) -> str:
        """Normalise a captured subject into an entity name.

        Deliberately not the object's normalisation. An object is a
        description — "song from the 1968 musical film featurette" — so
        cutting it at a modifier boundary leaves its head noun. A subject in
        this kind of prose is overwhelmingly a name, and the same cut damages
        one: "Deliver Us from Evil" becomes "Deliver Us" and "A Kiss for
        Corliss" becomes "Kiss". Stripping a leading determiner is wrong here
        for the same reason, and unnecessary — "The Laleli Mosque" and "Laleli
        Mosque" are joined by the resolver's containment rung.

        What is safe on this side is removing what is not part of any name: a
        parenthetical aside, and a predicate hanging off a copula.
        """
        cleaned = _PARENTHETICAL.sub(" ", str(token))

        # Once, not repeatedly: the cut is made at the first copula, which
        # removes every later one with it. "Redis has been located in memory"
        # gives the subject "Redis has been" and one cut leaves "Redis".
        #
        # A cut that leaves nothing is not refused. A subject that is all
        # predicate holds no entity, and the empty string it becomes is
        # rejected by _is_valid_statement, which drops the triple. Keeping the
        # uncut text instead would name an entity "is a historic mosque".
        copula = _COPULA.search(cleaned)
        if copula:
            cleaned = cleaned[: copula.start()]

        return self._clean(cleaned).strip("\"'“”()[]")

    def _normalize_object(self, token: str) -> str:
        """Normalise a captured object into the head noun phrase."""
        cleaned = self._clean(token).strip("\"'()[]")
        cleaned = _LEADING_DETERMINERS.sub("", cleaned)

        boundary = _MODIFIER_BOUNDARY.search(cleaned)
        if boundary:
            cleaned = cleaned[: boundary.start()]

        cleaned = self._truncate_on_word_boundary(cleaned)
        return _LEADING_DETERMINERS.sub("", cleaned).strip(" ,;")

    def _truncate_on_word_boundary(self, token: str) -> str:
        """Cut to MAX_OBJECT_LENGTH without splitting the final word."""
        if len(token) <= self.MAX_OBJECT_LENGTH:
            return token
        head = token[: self.MAX_OBJECT_LENGTH]
        if " " in head:
            head = head[: head.rindex(" ")]
        return head.strip()

    def _is_valid_statement(self, subject: str, relation: str, obj: str) -> bool:
        """Reject a triple that carries no usable entity on either side.

        This replaces a filter that discarded every ``is_a`` statement. That
        rule was aimed at objects like "song from the 1968 musical film
        featurette that keeps going", but it dropped a whole relation type to
        get at them. Cutting the object at its modifier boundary addresses the
        cause, so the relation can be kept.
        """
        if not subject or not obj:
            return False
        if obj.strip().lower() in _EMPTY_OBJECTS:
            return False
        if subject.strip().lower() == obj.strip().lower():
            return False
        return True

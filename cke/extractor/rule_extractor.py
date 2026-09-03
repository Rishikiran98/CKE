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

#: Where an appositive closes and the sentence resumes. "located in Istanbul,
#: is historic" gave the object "Istanbul, is historic": a comma is not a
#: modifier boundary, so the object ran to the end of the sentence. A comma
#: followed by a finite verb or a coordinator ends the phrase; a comma
#: followed by anything else does not, which keeps "Istanbul, Turkey" whole.
_APPOSITIVE_CLOSE = re.compile(
    r",\s+(?:is|was|are|were|has|have|had|and|but|which|who|that)\b", re.I
)

#: An object that reduces to one of these carries no entity.
_EMPTY_OBJECTS = frozenset(
    {"a", "an", "the", "it", "its", "this", "that", "these", "those", ""}
)

#: A subject that reduces to one of these names nothing. A pronoun stands for
#: an entity named in an earlier sentence, and resolving it needs coreference,
#: which this extractor does not do. Kept as a node it is worse than absent:
#: every document contributes its own facts to one entity called "He", and a
#: path through that node joins people who have nothing to do with each other.
#: The object side already refuses pronouns; this is the same rule on the
#: subject side.
_PRONOUN_SUBJECTS = frozenset(
    """he she it they we i you him her them his hers its their there this that
    these those who which what""".split()
)


class RuleExtractor:
    """Extract simple assertion patterns from free text."""

    MAX_OBJECT_LENGTH = 80

    #: Each entry pairs a sentence frame with the relation it yields. The
    #: relation is a format template over the match's named groups, so a frame
    #: that captures its own verb names the relation after it: one rule covers
    #: "directed by", "founded by" and "owned by" without listing any of them.
    #:
    #: Frames, not vocabulary. The previous four patterns named three verbs
    #: outright and matched the literal string " is a ", which missed "is an"
    #: and "was a" — 18.9% of the sentences it failed on contained a copula it
    #: could not read.
    PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
        # Copular predicate nominal: "X is an American director".
        (
            re.compile(
                r"(?P<s>[^.]+?)\s+(?:is|was|are|were)\s+(?:a|an|the)\s+(?P<o>[^.]+)",
                re.I,
            ),
            "is_a",
        ),
        # Passive with an agent: "X was directed by Y".
        (
            re.compile(
                r"(?P<s>[^.]+?)\s+(?:is|was|are|were|been)\s+"
                r"(?P<rel>\w{3,}ed)\s+by\s+(?P<o>[^.]+)",
                re.I,
            ),
            "{rel}_by",
        ),
        # Participle with a prepositional complement: "X was released in 1994",
        # "X, located in Istanbul, ...". The copula is optional, which is what
        # subsumes the former standalone located_in pattern.
        (
            re.compile(
                r"(?P<s>[^.]+?)\s+(?:(?:is|was|are|were|been)\s+)?"
                r"(?P<rel>\w{3,}ed)\s+(?P<prep>in|on|at)\s+(?P<o>[^.]+)",
                re.I,
            ),
            "{rel}_{prep}",
        ),
        # Active transitive, still named outright: no frame identifies a
        # transitive verb without a parser.
        (re.compile(r"(?P<s>[^.]+?)\s+uses\s+(?P<o>[^.]+)", re.I), "uses"),
        (re.compile(r"(?P<s>[^.]+?)\s+develops?\s+(?P<o>[^.]+)", re.I), "develops"),
    )

    def extract(self, text: str) -> list[Statement]:
        assertions: list[Statement] = []
        for sentence in self._split_sentences(text):
            for pattern, relation_template in self.PATTERNS:
                match = pattern.search(sentence)
                if not match:
                    continue
                relation = self._relation_for(relation_template, match)
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
    def _relation_for(template: str, match: re.Match[str]) -> str:
        """Fill a relation template from the frame's own captures."""
        captures = {
            name: value.lower()
            for name, value in match.groupdict().items()
            if value and name not in {"s", "o"}
        }
        return template.format(**captures) if captures else template

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

        for closer in (_MODIFIER_BOUNDARY, _APPOSITIVE_CLOSE):
            cut = closer.search(cleaned)
            if cut:
                cleaned = cleaned[: cut.start()]

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
        if subject.strip().lower() in _PRONOUN_SUBJECTS:
            return False
        if obj.strip().lower() in _EMPTY_OBJECTS:
            return False
        if subject.strip().lower() == obj.strip().lower():
            return False
        return True

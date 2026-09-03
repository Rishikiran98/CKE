"""Resolving a mention that names an entity a shorter way.

A source that writes "RESP" in one sentence and "RESP protocol" in the next
leaves two names for one thing. Fuzzy matching measures edit distance, and
those two strings differ by most of their characters, so the mention scored
far below the threshold and resolved to nothing.

The rule under test is containment with a refusal: a mention resolves to the
one candidate that contains it as a whole phrase, and to nothing at all when
more than one does.
"""

from __future__ import annotations

import pytest

from cke.entity_resolution.entity_resolver import EntityResolver


def _resolver(*entities: str) -> EntityResolver:
    resolver = EntityResolver()
    for entity in entities:
        resolver.register_alias(entity, entity)
    return resolver


def test_a_shorter_name_resolves_to_the_one_entity_containing_it():
    resolver = _resolver("RESP protocol", "Redis")

    assert resolver.resolve("RESP") == "RESP protocol"


def test_an_ambiguous_mention_resolves_to_no_entity():
    """Two containers is no evidence for either.

    "protocol" sits inside both names. There is nothing here to choose with,
    so the rule declines and leaves the mention to the fallback rather than
    attaching the query to whichever was seen first.
    """
    resolver = _resolver("RESP protocol", "text protocol")

    assert resolver.resolve("protocol") not in {"RESP protocol", "text protocol"}


def test_a_mention_of_only_uninformative_words_never_resolves():
    """ "The" is inside most entity names; a unique container is an accident."""
    resolver = _resolver("The Vermont Catamounts men's ice hockey team")

    assert resolver.resolve("the") != "The Vermont Catamounts men's ice hockey team"


def test_containment_is_a_phrase_not_a_token_subset():
    """The tokens must be consecutive.

    A subset test would match "Kansas City" against "Kansas has a city", which
    names something else entirely.
    """
    resolver = _resolver("Kansas has a city")

    assert resolver.resolve("Kansas City") != "Kansas has a city"


def test_a_phrase_inside_a_longer_name_still_resolves():
    resolver = _resolver("Kansas City jazz")

    assert resolver.resolve("Kansas City") == "Kansas City jazz"


def test_containment_does_not_override_an_exact_match():
    """A name that is itself an entity resolves to itself, not to a container."""
    resolver = _resolver("RESP", "RESP protocol")

    assert resolver.resolve("RESP") == "RESP"


def test_a_containment_match_reports_its_own_confidence():
    resolver = _resolver("RESP protocol")
    result = resolver.resolve_with_score("RESP")

    assert result.canonical == "RESP protocol"
    assert result.confidence == EntityResolver.CONTAINMENT_CONFIDENCE
    assert 0.50 < result.confidence < 0.75


def test_a_containment_match_is_not_cached_as_an_alias():
    """It is conditional on the candidate set, so it must be recomputed.

    A mention with one container today can have two once more text is
    ingested. Caching the first answer would hide that it had become
    ambiguous, and the rule's safety rests on ambiguity being detected.
    """
    resolver = _resolver("RESP protocol")
    assert resolver.resolve("RESP") == "RESP protocol"

    resolver.register_alias("RESP wire protocol", "RESP wire protocol")

    assert resolver.resolve("RESP") not in {"RESP protocol", "RESP wire protocol"}


@pytest.mark.parametrize("mention", ["", "   ", "of the"])
def test_an_empty_or_functional_mention_is_left_alone(mention):
    resolver = _resolver("RESP protocol")

    assert resolver.resolve(mention) != "RESP protocol"

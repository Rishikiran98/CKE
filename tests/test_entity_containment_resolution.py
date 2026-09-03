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


def test_an_ambiguous_mention_is_refused_by_the_containment_rung():
    """Two containers is no evidence for either.

    "protocol" sits inside both names. There is nothing here to choose with,
    so the rule declines rather than attaching the query to whichever was seen
    first.

    Asserted against the rung and not against ``resolve``, because the chain
    does not stop at a refusal: the rung below is embedding similarity, which
    without sentence-transformers compares hashed token counts against a
    threshold its own degradation message says was not calibrated for them. It
    answers "Kansas City" with "Kansas City Chiefs" at 0.82 where this rung
    declined. Refusing ambiguity is this rung's contract; it is not yet the
    chain's.
    """
    resolver = _resolver("RESP protocol", "text protocol")
    candidates = resolver._graph_candidates()

    assert resolver._unique_container("protocol", candidates) is None


def test_the_chain_does_not_return_a_container_it_refused():
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


# ----------------------------------------------------------------------
# Fan-out: one mention to several entities
# ----------------------------------------------------------------------


def test_an_ambiguous_mention_expands_to_every_container():
    """What resolve() must refuse, retrieval wants all of.

    Picking one entity and starting a walk there needs a decision. Choosing
    where in the graph to start does not: both are starting points, and path
    scoring is what ranks them.
    """
    resolver = _resolver("Kansas City jazz", "Kansas City Chiefs", "Redis")

    assert set(resolver.expand(["Kansas City"])) == {
        "Kansas City jazz",
        "Kansas City Chiefs",
    }


def test_expansion_is_ordered_by_how_tightly_the_mention_fits():
    """The mention covers half of one name and a fifth of the other.

    The tight fit is named so that it sorts *last* alphabetically: an
    implementation that lost the ranking and fell back to name order would
    return these the other way round.
    """
    resolver = _resolver("a RESP wire protocol format frame", "zeta RESP")

    assert resolver.expand(["RESP"]) == [
        "zeta RESP",
        "a RESP wire protocol format frame",
    ]


def test_expansion_is_a_phrase_match_not_a_token_subset():
    resolver = _resolver("Kansas has a city")

    assert "Kansas has a city" not in resolver.expand(["Kansas City"])


def test_a_mention_that_matches_an_entity_exactly_expands_to_it():
    resolver = _resolver("Redis")

    assert resolver.expand(["Redis"]) == ["Redis"]


def test_a_mention_that_expands_to_nothing_keeps_its_resolved_form():
    """Expansion adds starting points; it never removes the one already there.

    No entity holds "Postgres", so there is nothing to fan out to, and the
    mention must still arrive at the retriever as whatever resolve() makes of
    it rather than being dropped on the floor.
    """
    resolver = _resolver("Redis")
    expanded = resolver.expand(["Postgres"])

    assert expanded == [resolver.resolve("Postgres")]
    assert expanded != []


def test_expansion_is_capped_per_mention():
    resolver = _resolver(*[f"Redis node {i}" for i in range(10)])

    assert len(resolver.expand(["Redis"])) == EntityResolver.EXPANSION_PER_MENTION


def test_expansion_is_capped_in_total_across_mentions():
    """Three mentions, three distinct entities each: nine before the total cap."""
    resolver = _resolver(
        *[f"{stem} {i}" for stem in ("alpha", "bravo", "charlie") for i in range(3)]
    )
    expanded = resolver.expand(["alpha", "bravo", "charlie"])

    assert len(set(expanded)) == len(expanded)
    assert len(expanded) == EntityResolver.EXPANSION_TOTAL


def test_expansion_deduplicates_across_mentions():
    resolver = _resolver("RESP protocol")

    assert resolver.expand(["RESP", "protocol"]) == ["RESP protocol"]


def test_expansion_is_reproducible():
    """Ties break on the name, so two runs seed the same walk."""
    entities = ["Redis alpha", "Redis bravo", "Redis charlie"]
    first = _resolver(*entities).expand(["Redis"])
    second = _resolver(*reversed(entities)).expand(["Redis"])

    assert first == second

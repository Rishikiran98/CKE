from cke.entity_resolution.entity_resolver import EntityResolver, ResolutionResult
from cke.graph_engine.graph_engine import KnowledgeGraphEngine
from cke.models import Statement

# ------------------------------------------------------------------
# Existing tests (preserved)
# ------------------------------------------------------------------


def test_resolve_entity_aliases():
    resolver = EntityResolver(
        aliases={"redis db": "Redis", "remote dictionary server": "Redis"}
    )
    assert resolver.resolve_entity("Redis DB") == "Redis"
    assert resolver.resolve_entity("Remote Dictionary Server") == "Redis"


def test_merge_entities():
    resolver = EntityResolver()
    resolver.resolve_entity("Redis")
    resolver.resolve_entity("Redis Database")
    canonical = resolver.merge_entities("Redis", "Redis Database")
    assert canonical in {"Redis", "Redis Database"}
    assert resolver.resolve_entity("Redis Database") == canonical


def test_canonical_mapping_variants_resolve_to_same_entity():
    resolver = EntityResolver()
    canonical = resolver.resolve_entity("Redis")
    assert resolver.resolve_entity("Redis DB") == canonical
    assert resolver.resolve_entity("Redis database") == canonical


# ------------------------------------------------------------------
# New: resolve_with_score confidence tiers
# ------------------------------------------------------------------


def test_resolve_with_score_exact_canonical():
    resolver = EntityResolver()
    resolver.register_alias("Redis", "Redis")
    result = resolver.resolve_with_score("Redis")
    assert result.canonical == "Redis"
    assert result.confidence >= 0.90


def test_resolve_with_score_alias():
    resolver = EntityResolver(aliases={"redis db": "Redis"})
    result = resolver.resolve_with_score("Redis DB")
    assert result.canonical == "Redis"
    assert result.confidence >= 0.85


def test_resolve_with_score_normalised_match():
    resolver = EntityResolver()
    resolver.register_alias("Redis", "Redis")
    result = resolver.resolve_with_score("redis")
    assert result.canonical == "Redis"
    assert result.confidence >= 0.70


# ------------------------------------------------------------------
# New: lowercase resolution
# ------------------------------------------------------------------


def test_lowercase_mention_resolves():
    resolver = EntityResolver()
    resolver.register_alias("Albert Einstein", "Albert Einstein")
    assert resolver.resolve("albert einstein") == "Albert Einstein"


# ------------------------------------------------------------------
# New: partial name via fuzzy matching
# ------------------------------------------------------------------


def test_partial_name_fuzzy_match():
    graph = KnowledgeGraphEngine()
    graph.add_statement("Albert Einstein", "born_in", "Ulm")
    resolver = EntityResolver(graph_engine=graph, fuzzy_threshold=0.5)
    result = resolver.resolve_with_score("Einstein")
    assert result.canonical == "Albert Einstein"
    assert result.confidence > 0.50


# ------------------------------------------------------------------
# New: extract_entities with graph clues
# ------------------------------------------------------------------


def test_extract_entities_finds_graph_entities():
    graph = KnowledgeGraphEngine()
    graph.add_statement("Redis", "supports", "PubSub")
    resolver = EntityResolver(graph_engine=graph)
    entities = resolver.extract_entities("Tell me about Redis")
    assert "Redis" in entities


def test_extract_entities_capitalised_phrases():
    resolver = EntityResolver()
    entities = resolver.extract_entities(
        "How are Scott Derrickson and Ed Wood connected?"
    )
    assert "Scott Derrickson" in entities
    assert "Ed Wood" in entities


def test_extract_entities_context_clue_matching():
    graph = KnowledgeGraphEngine()
    graph.add_statements(
        [
            Statement("Redis", "supports", "PubSub"),
            Statement("Postgres", "supports", "SQL"),
        ]
    )
    resolver = EntityResolver(graph_engine=graph)
    entities = resolver.extract_entities("Which database supports pubsub?")
    assert "Redis" in entities


# ------------------------------------------------------------------
# New: resolve() and canonicalize() aliases
# ------------------------------------------------------------------


def test_resolve_alias_method():
    resolver = EntityResolver(aliases={"redis db": "Redis"})
    assert resolver.resolve("Redis DB") == "Redis"
    assert resolver.canonicalize("Redis DB") == "Redis"


# ------------------------------------------------------------------
# New: ResolutionResult dataclass
# ------------------------------------------------------------------


def test_resolution_result_dataclass():
    r = ResolutionResult(canonical="Redis", confidence=0.95)
    assert r.canonical == "Redis"
    assert r.confidence == 0.95

from cke.entity_resolution.entity_resolver import EntityResolver


def test_resolve_entity_aliases():
    resolver = EntityResolver(aliases={"redis db": "Redis", "remote dictionary server": "Redis"})
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

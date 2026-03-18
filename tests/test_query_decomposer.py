from cke.router.query_decomposer import QueryDecomposer


def test_query_decomposer_hotpot_style_chain():
    decomposer = QueryDecomposer()
    query = (
        "What nationality is the director of the film that starred Tom Cruise "
        "in Top Gun?"
    )
    result = decomposer.decompose(query, entities=["Top Gun", "Tom Cruise"])

    values = [(step.step_type, step.value) for step in result.steps]
    assert ("entity", "Top Gun") in values
    assert ("entity", "Tom Cruise") in values
    assert ("relation", "starred_in") in values
    assert ("relation", "directed_by") in values
    assert ("relation", "nationality") in values


def test_query_decomposer_sets_multi_hop_bridge_hints():
    decomposer = QueryDecomposer()

    result = decomposer.decompose(
        "Which film is associated with the character portrayed by Person X?",
        entities=["Person X"],
    )

    assert result.multi_hop_hint is True
    assert result.bridge_entities_expected is True

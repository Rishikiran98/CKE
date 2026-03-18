"""Config helpers for transparent retrieval and ranking heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(slots=True)
class ChunkRankingWeights:
    dense_weight: float = 0.55
    entity_overlap_weight: float = 0.2
    relation_overlap_weight: float = 0.15
    alias_overlap_weight: float = 0.1
    source_trust_bonus: float = 0.05


@dataclass(slots=True)
class FactRankingWeights:
    chunk_weight: float = 0.35
    entity_weight: float = 0.2
    relation_weight: float = 0.22
    trust_weight: float = 0.1
    canonical_match_bonus: float = 0.08
    relation_target_bonus: float = 0.14
    operator_bonus: float = 0.12


@dataclass(slots=True)
class PathRankingWeights:
    entity_weight: float = 0.28
    relation_weight: float = 0.24
    trust_weight: float = 0.18
    direct_bridge_bonus: float = 0.12
    multi_hop_bonus: float = 0.1
    comparison_bonus: float = 0.08
    diversity_bonus: float = 0.04
    length_penalty: float = 0.08


@dataclass(slots=True)
class RetrievalRankingConfig:
    chunk: ChunkRankingWeights = field(default_factory=ChunkRankingWeights)
    fact: FactRankingWeights = field(default_factory=FactRankingWeights)
    path: PathRankingWeights = field(default_factory=PathRankingWeights)


def load_ranking_config(
    config_path: str | Path | None = "configs/retrieval_ranking.yaml",
) -> RetrievalRankingConfig:
    """Load heuristic ranking weights from YAML, falling back to sane defaults."""
    config = RetrievalRankingConfig()
    if config_path is None:
        return config
    path = Path(config_path)
    if not path.exists():
        return config

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    chunk = payload.get("chunk", {}) or {}
    fact = payload.get("fact", {}) or {}
    path_cfg = payload.get("path", {}) or {}

    return RetrievalRankingConfig(
        chunk=ChunkRankingWeights(
            dense_weight=float(chunk.get("dense_weight", config.chunk.dense_weight)),
            entity_overlap_weight=float(
                chunk.get("entity_overlap_weight", config.chunk.entity_overlap_weight)
            ),
            relation_overlap_weight=float(
                chunk.get(
                    "relation_overlap_weight", config.chunk.relation_overlap_weight
                )
            ),
            alias_overlap_weight=float(
                chunk.get("alias_overlap_weight", config.chunk.alias_overlap_weight)
            ),
            source_trust_bonus=float(
                chunk.get("source_trust_bonus", config.chunk.source_trust_bonus)
            ),
        ),
        fact=FactRankingWeights(
            chunk_weight=float(fact.get("chunk_weight", config.fact.chunk_weight)),
            entity_weight=float(fact.get("entity_weight", config.fact.entity_weight)),
            relation_weight=float(
                fact.get("relation_weight", config.fact.relation_weight)
            ),
            trust_weight=float(fact.get("trust_weight", config.fact.trust_weight)),
            canonical_match_bonus=float(
                fact.get("canonical_match_bonus", config.fact.canonical_match_bonus)
            ),
            relation_target_bonus=float(
                fact.get("relation_target_bonus", config.fact.relation_target_bonus)
            ),
            operator_bonus=float(
                fact.get("operator_bonus", config.fact.operator_bonus)
            ),
        ),
        path=PathRankingWeights(
            entity_weight=float(
                path_cfg.get("entity_weight", config.path.entity_weight)
            ),
            relation_weight=float(
                path_cfg.get("relation_weight", config.path.relation_weight)
            ),
            trust_weight=float(path_cfg.get("trust_weight", config.path.trust_weight)),
            direct_bridge_bonus=float(
                path_cfg.get("direct_bridge_bonus", config.path.direct_bridge_bonus)
            ),
            multi_hop_bonus=float(
                path_cfg.get("multi_hop_bonus", config.path.multi_hop_bonus)
            ),
            comparison_bonus=float(
                path_cfg.get("comparison_bonus", config.path.comparison_bonus)
            ),
            diversity_bonus=float(
                path_cfg.get("diversity_bonus", config.path.diversity_bonus)
            ),
            length_penalty=float(
                path_cfg.get("length_penalty", config.path.length_penalty)
            ),
        ),
    )

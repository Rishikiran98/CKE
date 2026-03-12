"""Assertion deduplication helpers."""

from __future__ import annotations

import hashlib
import json

from cke.graph.assertion import Assertion
from cke.graph.trust_engine import TrustEngine


class AssertionDeduplicator:
    """Merge duplicate assertions by identity + qualifier bucket."""

    def __init__(self, trust_engine: TrustEngine | None = None) -> None:
        self.trust_engine = trust_engine or TrustEngine()

    @staticmethod
    def _qualifier_bucket(qualifiers: dict) -> str:
        return json.dumps(qualifiers or {}, sort_keys=True, separators=(",", ":"))

    def assertion_hash(self, assertion: Assertion) -> str:
        span_text = "||".join(
            sorted((e.text or "") for e in assertion.evidence if getattr(e, "text", ""))
        )
        raw = "||".join(
            [
                assertion.subject,
                assertion.relation,
                assertion.object,
                span_text,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def deduplicate(self, assertions: list[Assertion]) -> list[Assertion]:
        merged: dict[str, Assertion] = {}
        for assertion in assertions:
            key = self.assertion_hash(assertion)
            if key not in merged:
                merged[key] = assertion
                continue

            existing = merged[key]
            existing.evidence_count += max(assertion.evidence_count, 1)
            existing.evidence.extend(assertion.evidence)
            existing.extractor_confidence = max(
                existing.extractor_confidence,
                assertion.extractor_confidence,
            )
            if assertion.timestamp > existing.timestamp:
                existing.timestamp = assertion.timestamp
            self.trust_engine.compute_trust(existing)

        return list(merged.values())

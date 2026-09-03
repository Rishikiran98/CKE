"""Domain classification helpers for entities and assertions."""

from __future__ import annotations

from collections.abc import Iterable

from cke.diagnostics import DegradationMixin
from cke.graph.assertion import Assertion


#: Label applied when nothing matched. Not a classification.
_UNCLASSIFIED_DOMAIN = "programming"


class DomainClassifier(DegradationMixin):
    """Assign domain labels to entities/assertions via lightweight heuristics."""

    DOMAIN_KEYWORDS: dict[str, set[str]] = {
        "databases": {
            "database",
            "sql",
            "nosql",
            "postgres",
            "mysql",
            "redis",
            "mongodb",
            "index",
            "query",
        },
        "cloud": {
            "aws",
            "azure",
            "gcp",
            "kubernetes",
            "docker",
            "cloud",
            "serverless",
            "s3",
        },
        "programming": {
            "python",
            "java",
            "rust",
            "javascript",
            "function",
            "class",
            "compiler",
            "algorithm",
        },
        "biology": {
            "cell",
            "gene",
            "protein",
            "species",
            "dna",
            "rna",
            "organism",
        },
        "history": {
            "empire",
            "war",
            "century",
            "ancient",
            "medieval",
            "dynasty",
            "revolution",
        },
    }

    def __init__(
        self, embedding_backend: object | None = None, strict: bool = False
    ) -> None:
        self._init_degradation(strict)
        self.embedding_backend = embedding_backend

    def _keyword_score(self, text: str) -> dict[str, int]:
        tokens = set(text.lower().replace("_", " ").replace("-", " ").split())
        return {
            domain: len(tokens.intersection(keywords))
            for domain, keywords in self.DOMAIN_KEYWORDS.items()
        }

    def _embedding_hint(self, text: str) -> str | None:
        """Optional embedding classifier hook.

        If an embedding backend is provided and has `predict_domain(text)`,
        use it as a fallback when keyword scores tie or are all zero.
        """
        if self.embedding_backend and hasattr(self.embedding_backend, "predict_domain"):
            return self.embedding_backend.predict_domain(text)
        return None

    def classify_entity(self, entity_name: str) -> str:
        """Classify an entity, or declare that it could not be classified.

        No keyword matched and no embedding hint is available, so the result
        used to be a real domain label indistinguishable from a match.
        """
        scores = self._keyword_score(entity_name)
        best_domain, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score > 0:
            return best_domain

        hint = self._embedding_hint(entity_name)
        if hint:
            return hint

        self._degrade(
            f"no keyword matched {entity_name!r} and no embedding hint is "
            f"available, so it is labelled {_UNCLASSIFIED_DOMAIN!r}, which is "
            "indistinguishable downstream from a real classification"
        )
        return _UNCLASSIFIED_DOMAIN

    def classify_assertion(self, assertion: Assertion) -> str:
        content = " ".join(
            [assertion.subject, assertion.relation, assertion.object]
            + [str(value) for value in assertion.qualifiers.values()]
        )
        return self.classify_entity(content)

    def classify_entities(self, entity_names: Iterable[str]) -> dict[str, str]:
        return {name: self.classify_entity(name) for name in entity_names}

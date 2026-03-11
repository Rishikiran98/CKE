"""Heuristic query intent classifier."""

from __future__ import annotations


class IntentClassifier:
    """Classify queries into retrieval intents using keyword rules."""

    def classify(self, query: str) -> str:
        q = query.lower()

        if any(
            token in q for token in ["verify", "true", "false", "correct", "confirm"]
        ):
            return "verification"
        if any(token in q for token in ["define", "what is", "meaning of", "explain"]):
            return "definition"
        if any(
            token in q
            for token in ["compare", "difference", "better than", "vs", "versus"]
        ):
            return "comparison"
        if any(token in q for token in ["which", "uses", "supports", "through", "via"]):
            return "multi-hop"
        return "factoid"

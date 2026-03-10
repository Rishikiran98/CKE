"""Template-based reasoning from graph statements."""

from __future__ import annotations

from typing import List

from cke.models import Statement


class TemplateReasoner:
    """Simple deterministic reasoner over retrieved statements."""

    def answer(self, query: str, context: List[Statement]) -> str:
        if not context:
            return "I don't have enough graph context to answer that yet."

        # Heuristic for "what protocol" style queries.
        if "protocol" in query.lower():
            for st in context:
                if st.relation in {"uses", "implemented_via"}:
                    return f"{st.subject} uses {st.object} protocol."

        if len(context) >= 2:
            a, b = context[0], context[1]
            if a.object == b.subject:
                return f"{a.subject} {a.relation} {a.object}, and {a.object} {b.relation} {b.object}."

        top = context[0]
        return f"{top.subject} {top.relation} {top.object}."

    def format_reasoning_path(self, context: List[Statement]) -> str:
        if not context:
            return "No path found."
        return "\n".join(f"{st.subject} -> {st.relation} -> {st.object}" for st in context)

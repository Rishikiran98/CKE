"""Utilities for summarizing per-domain convergence drift."""

from __future__ import annotations

from cke.graph.domain_registry import DomainRegistry


def domain_drift_summary(graph) -> dict[str, dict[str, float | str]]:
    registry: DomainRegistry | None = getattr(graph, "domain_registry", None)
    if registry is None:
        return {}

    summary: dict[str, dict[str, float | str]] = {}
    for domain_name, record in registry._records.items():  # noqa: SLF001
        summary[domain_name] = {
            "state": record.state,
            "drift_score": record.drift_score,
            "last_update": record.last_update,
        }
    return summary


def stable_domains(graph) -> list[str]:
    summary = domain_drift_summary(graph)
    return sorted(name for name, stats in summary.items() if stats["state"] == "stable")


def volatile_domains(graph) -> list[str]:
    summary = domain_drift_summary(graph)
    return sorted(
        name for name, stats in summary.items() if stats["state"] == "volatile"
    )

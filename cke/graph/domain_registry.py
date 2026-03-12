"""Registry for current per-domain convergence state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from cke.graph.convergence_engine import ConvergenceEngine


@dataclass(slots=True)
class DomainRecord:
    domain_name: str
    state: str
    last_update: str
    drift_score: float


class DomainRegistry:
    """Persist and query convergence state for each domain."""

    def __init__(self, convergence_engine: ConvergenceEngine | None = None) -> None:
        self.convergence_engine = convergence_engine or ConvergenceEngine()
        self._records: dict[str, DomainRecord] = {}

    def update_domain(self, domain_name: str, drift_score: float) -> DomainRecord:
        state = self.convergence_engine.update(domain_name, drift_score)
        record = DomainRecord(
            domain_name=domain_name,
            state=state,
            last_update=datetime.now(timezone.utc).isoformat(),
            drift_score=drift_score,
        )
        self._records[domain_name] = record
        return record

    def get_domain_state(self, domain_name: str) -> str | None:
        record = self._records.get(domain_name)
        return record.state if record else None

    def get_record(self, domain_name: str) -> DomainRecord | None:
        return self._records.get(domain_name)

"""Environment diagnostics and the degradation contract.

CKE has optional dependencies and components that can run with reduced
capability when one is missing. A benchmark that quietly ran on a hashed
bag-of-words embedder instead of a sentence transformer is not a benchmark, so
degradation here is never silent. Every degrading component must:

1. Emit a ``WARNING`` log naming the specific cause.
2. Set an inspectable flag on the object (``degraded`` / ``degraded_reason``).
3. Raise :class:`DegradedComponentError` instead of degrading when it was
   constructed with ``strict=True``.

:class:`DegradationMixin` implements all three. Components inherit it and call
``_init_degradation`` in ``__init__``, then ``_degrade(reason)`` wherever they
would otherwise fall back.

:func:`environment_report` answers "what is actually running": which optional
dependencies resolved and at what version, which models loaded, and which
components have degraded so far in this process. Evaluation, benchmark, and
experiment entry points print it before reporting any number.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import platform
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "DegradedComponentError",
    "DegradationMixin",
    "DegradationRecord",
    "DependencyStatus",
    "EnvironmentReport",
    "LoadedModel",
    "OPTIONAL_DEPENDENCIES",
    "clear_runtime_state",
    "declare_degradation",
    "environment_report",
    "record_degradation",
    "record_loaded_model",
]

logger = logging.getLogger(__name__)


class DegradedComponentError(RuntimeError):
    """Raised when a component would degrade but was constructed strict.

    The message names the component and the specific cause, so a failed
    evaluation run says what was missing rather than producing a number.
    """


@dataclass(frozen=True)
class DegradationRecord:
    """One component that degraded during this process."""

    component: str
    reason: str


@dataclass(frozen=True)
class LoadedModel:
    """A model a component actually loaded, as opposed to asked for."""

    component: str
    requested: str
    loaded: str


@dataclass(frozen=True)
class DependencyStatus:
    """Whether one optional dependency resolved, and at what version."""

    import_name: str
    distribution: str
    purpose: str
    available: bool
    version: str | None = None
    error: str | None = None


# Every optional third-party import guarded by a try/except in cke/.
# (import name, distribution name, what it is used for)
OPTIONAL_DEPENDENCIES: tuple[tuple[str, str, str], ...] = (
    ("sentence_transformers", "sentence-transformers", "dense text embeddings"),
    ("faiss", "faiss-cpu", "vector index for dense retrieval"),
    ("openai", "openai", "LLM extraction and reasoning"),
    ("numpy", "numpy", "vector arithmetic"),
    ("networkx", "networkx", "in-memory graph topology"),
    ("neo4j", "neo4j", "Neo4j graph backend"),
    ("fastapi", "fastapi", "HTTP API service"),
    ("yaml", "PyYAML", "YAML configuration parsing"),
    ("rapidfuzz", "rapidfuzz", "fuzzy string matching for entity resolution"),
    ("spacy", "spacy", "coreference resolution"),
)


# Runtime state, populated by components as they are constructed and used.
_DEGRADATIONS: list[DegradationRecord] = []
_LOADED_MODELS: list[LoadedModel] = []


def record_degradation(component: str, reason: str) -> None:
    """Record that *component* degraded, so the report can surface it."""
    record = DegradationRecord(component=component, reason=reason)
    if record not in _DEGRADATIONS:
        _DEGRADATIONS.append(record)


def record_loaded_model(component: str, requested: str, loaded: str) -> None:
    """Record the model identity a component actually loaded."""
    record = LoadedModel(component=component, requested=requested, loaded=loaded)
    if record not in _LOADED_MODELS:
        _LOADED_MODELS.append(record)


def clear_runtime_state() -> None:
    """Forget recorded degradations and models. For tests and repeated runs."""
    _DEGRADATIONS.clear()
    _LOADED_MODELS.clear()


def declare_degradation(component: str, reason: str, strict: bool = False) -> None:
    """Declare a degradation from code that has no object to flag.

    Same contract as :meth:`DegradationMixin._degrade`: warn, record, and raise
    under ``strict``. Use it in module-level functions such as config loaders.
    """
    if strict:
        raise DegradedComponentError(
            f"{component} would run degraded: {reason}. "
            f"It was called with strict=True, which forbids that. "
            f"Install or configure the missing component, or call it with "
            f"strict=False to accept degraded behaviour."
        )
    logger.warning("%s is running degraded: %s", component, reason)
    record_degradation(component, reason)


class DegradationMixin:
    """Give a component the three-part degradation contract.

    Subclasses call ``_init_degradation(strict)`` from ``__init__`` before any
    work that might degrade, then ``_degrade(reason)`` at each fallback.
    """

    strict: bool
    degraded: bool
    degraded_reason: str

    def _init_degradation(self, strict: bool = False) -> None:
        """Initialise degradation state. Call first in ``__init__``."""
        self.strict = bool(strict)
        self.degraded = False
        self.degraded_reason = ""

    def _degrade(self, reason: str) -> None:
        """Declare that this component is running with reduced capability.

        Logs a warning, marks the object, and raises when strict.
        """
        component = type(self).__name__
        if getattr(self, "strict", False):
            raise DegradedComponentError(
                f"{component} would run degraded: {reason}. "
                f"It was constructed with strict=True, which forbids that. "
                f"Install or configure the missing component, or construct it "
                f"with strict=False to accept degraded behaviour."
            )

        # Keep every distinct reason, but never repeat one. A degradation
        # reached from inside a loop would otherwise emit a warning per
        # iteration and grow degraded_reason quadratically.
        existing = getattr(self, "degraded_reason", "")
        reasons = existing.split("; ") if existing else []
        if reason in reasons:
            return

        self.degraded = True
        reasons.append(reason)
        self.degraded_reason = "; ".join(reasons)
        logger.warning("%s is running degraded: %s", component, reason)
        record_degradation(component, reason)


def _probe(import_name: str, distribution: str, purpose: str) -> DependencyStatus:
    """Import one optional dependency and report what happened."""
    try:
        importlib.import_module(import_name)
    except ImportError as exc:
        return DependencyStatus(
            import_name=import_name,
            distribution=distribution,
            purpose=purpose,
            available=False,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 - a broken install is not an absence
        # The package exists but blew up on import. That is a different
        # problem from it being missing, and the report must not conflate them.
        return DependencyStatus(
            import_name=import_name,
            distribution=distribution,
            purpose=purpose,
            available=False,
            error=f"{type(exc).__name__} during import: {exc}",
        )

    version: str | None
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = None

    return DependencyStatus(
        import_name=import_name,
        distribution=distribution,
        purpose=purpose,
        available=True,
        version=version,
    )


@dataclass
class EnvironmentReport:
    """What is actually running, as opposed to what was requested."""

    python_version: str
    platform: str
    dependencies: list[DependencyStatus] = field(default_factory=list)
    loaded_models: list[LoadedModel] = field(default_factory=list)
    degradations: list[DegradationRecord] = field(default_factory=list)

    @property
    def missing(self) -> list[DependencyStatus]:
        """Optional dependencies that did not resolve."""
        return [dep for dep in self.dependencies if not dep.available]

    @property
    def is_degraded(self) -> bool:
        """True if any component has reported degraded operation."""
        return bool(self.degradations)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form, for embedding in results metadata."""
        return {
            "python_version": self.python_version,
            "platform": self.platform,
            "dependencies": [
                {
                    "import_name": dep.import_name,
                    "distribution": dep.distribution,
                    "purpose": dep.purpose,
                    "available": dep.available,
                    "version": dep.version,
                    "error": dep.error,
                }
                for dep in self.dependencies
            ],
            "loaded_models": [
                {
                    "component": model.component,
                    "requested": model.requested,
                    "loaded": model.loaded,
                }
                for model in self.loaded_models
            ],
            "degradations": [
                {"component": record.component, "reason": record.reason}
                for record in self.degradations
            ],
        }

    def render(self) -> str:
        """Return the report as a human-readable block."""
        lines: list[str] = []
        lines.append("=" * 72)
        lines.append("CKE environment report")
        lines.append("=" * 72)
        lines.append(f"Python:   {self.python_version}")
        lines.append(f"Platform: {self.platform}")
        lines.append("")
        lines.append("Optional dependencies:")
        for dep in self.dependencies:
            if dep.available:
                version = dep.version or "unknown version"
                lines.append(
                    f"  [ok]      {dep.distribution} {version} ({dep.purpose})"
                )
            else:
                lines.append(
                    f"  [MISSING] {dep.distribution} ({dep.purpose}): {dep.error}"
                )

        lines.append("")
        if self.loaded_models:
            lines.append("Models loaded:")
            for model in self.loaded_models:
                if model.requested == model.loaded:
                    lines.append(f"  [ok]      {model.component}: {model.loaded}")
                else:
                    lines.append(
                        f"  [CHANGED] {model.component}: requested "
                        f"{model.requested}, loaded {model.loaded}"
                    )
        else:
            lines.append("Models loaded: none yet")

        lines.append("")
        if self.degradations:
            lines.append("DEGRADED COMPONENTS — results from this run are not valid:")
            for record in self.degradations:
                lines.append(f"  [DEGRADED] {record.component}: {record.reason}")
        else:
            lines.append("Degraded components: none")
        lines.append("=" * 72)
        return "\n".join(lines)


def environment_report(
    dependencies: Iterable[tuple[str, str, str]] | None = None,
) -> EnvironmentReport:
    """Report which optional dependencies resolved and what has degraded.

    Probing imports each optional dependency, so call it once at the start of a
    run rather than in a loop.
    """
    specs = tuple(dependencies) if dependencies is not None else OPTIONAL_DEPENDENCIES
    return EnvironmentReport(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        dependencies=[_probe(*spec) for spec in specs],
        loaded_models=list(_LOADED_MODELS),
        degradations=list(_DEGRADATIONS),
    )

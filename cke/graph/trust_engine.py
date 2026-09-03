"""Trust scoring utilities for assertions.

Calibration weights come from a YAML file. When that file is missing or
unreadable the engine can still run on built-in defaults, but every trust score
it then produces was computed with weights nobody chose, so the substitution is
declared rather than made quietly.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from cke.diagnostics import DegradationMixin
from cke.graph.assertion import Assertion
from cke.trust.calibration import TrustCalibrationConfig, TrustCalibrator

try:
    import yaml
except ImportError:  # pragma: no cover - optional runtime dependency
    yaml = None


_CONFIG_KEYS = ("w_src", "w_freq", "w_conf", "tau", "low_trust_threshold")


class TrustEngine(DegradationMixin):
    """Compute trust scores from source quality, evidence, and recency.

    Args:
        tau: recency decay constant. Overrides the configured value when given;
            leave as None to use whatever the config file specifies.
        source_weights: per-source multipliers.
        calibrator: preconstructed calibrator, bypassing config loading.
        config_path: YAML calibration file, or None to use built-in defaults
            deliberately. Note this default is relative to the working
            directory, so running from elsewhere will report a degradation.
        strict: when True, raise rather than substitute built-in defaults.
    """

    def __init__(
        self,
        tau: float | None = None,
        source_weights: dict[str, float] | None = None,
        calibrator: TrustCalibrator | None = None,
        config_path: str | Path | None = "configs/trust_config.yaml",
        strict: bool = False,
    ) -> None:
        self._init_degradation(strict)
        self.source_weights = source_weights or {
            "wikipedia": 1.0,
            "paper": 1.2,
            "docs": 1.1,
            "unknown": 0.7,
        }
        if calibrator is not None:
            # The supplied calibrator carries its own configuration, so
            # loading one here would only be able to fail over a file whose
            # values are never used.
            self.calibrator = calibrator
            return

        calibration = self._load_config(config_path)
        if tau is not None:
            calibration.tau = tau
        self.calibrator = TrustCalibrator(config=calibration)

    def _load_config(self, config_path: str | Path | None) -> TrustCalibrationConfig:
        """Load calibration weights from YAML.

        Passing ``config_path=None`` is an explicit choice to use defaults and
        is not a degradation. Anything else that prevents the file being read
        is.
        """
        cfg = TrustCalibrationConfig()
        if config_path is None:
            return cfg

        path = Path(config_path)
        if not path.exists():
            self._degrade(
                f"trust calibration config {path} does not exist (resolved "
                f"from {Path.cwd()}), so built-in default weights are used and "
                "every trust score reflects defaults rather than configuration"
            )
            return cfg

        if yaml is None:
            self._degrade(
                f"PyYAML is not installed, so {path} cannot be parsed and "
                "built-in default trust weights are used instead. Install it "
                "with `pip install PyYAML`"
            )
            return cfg

        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            self._degrade(
                f"trust calibration config {path} could not be parsed "
                f"({type(exc).__name__}: {exc}), so built-in default weights "
                "are used"
            )
            return cfg

        if not isinstance(payload, dict):
            self._degrade(
                f"trust calibration config {path} is not a mapping, so "
                "built-in default weights are used"
            )
            return cfg

        data: dict[str, float] = {}
        for key in _CONFIG_KEYS:
            if key not in payload:
                continue
            try:
                data[key] = float(payload[key])
            except (TypeError, ValueError):
                self._degrade(
                    f"trust calibration key {key!r} in {path} is not a number "
                    f"({payload[key]!r}), so the built-in default for it is used"
                )

        unknown = sorted(set(payload) - set(_CONFIG_KEYS))
        if unknown:
            self._degrade(
                f"trust calibration config {path} contains keys this version "
                f"does not understand ({', '.join(unknown)}); they are ignored"
            )

        return TrustCalibrationConfig(
            w_src=data.get("w_src", cfg.w_src),
            w_freq=data.get("w_freq", cfg.w_freq),
            w_conf=data.get("w_conf", cfg.w_conf),
            tau=data.get("tau", cfg.tau),
            low_trust_threshold=data.get(
                "low_trust_threshold", cfg.low_trust_threshold
            ),
        )

    def compute_trust(self, assertion: Assertion, now: float | None = None) -> float:
        """Compute ingestion trust from source, extraction confidence, and decay."""
        source_weight = self.source_weights.get(
            assertion.source, self.source_weights["unknown"]
        )
        now_ts = float(time.time()) if now is None else float(now)
        observed = float(assertion.timestamp)
        age = max(0.0, now_ts - observed)
        tau = max(float(self.calibrator.config.tau), 1e-9)
        temporal_decay = math.exp(-(age / tau))
        extraction_confidence = max(
            0.0, min(1.0, float(assertion.extractor_confidence))
        )

        trust = max(
            0.0, min(1.0, source_weight * extraction_confidence * temporal_decay)
        )
        assertion.trust_score = trust
        assertion.confidence = trust
        return trust

    def fit_from_graph(self, graph: object) -> dict[str, float]:
        """Run batch calibration from graph statistics."""
        return self.calibrator.fit_from_graph(graph)

    def trust_distribution_stats(
        self,
        assertions: list[Assertion],
    ) -> dict[str, float]:
        """Return aggregate trust distribution statistics."""
        if not assertions:
            return {"mean_trust": 0.0, "variance": 0.0, "low_trust_ratio": 0.0}
        scores = [float(item.trust_score) for item in assertions]
        mean = sum(scores) / len(scores)
        variance = sum((value - mean) ** 2 for value in scores) / len(scores)
        low_cutoff = self.calibrator.config.low_trust_threshold
        low_ratio = sum(1 for score in scores if score < low_cutoff) / len(scores)
        return {
            "mean_trust": mean,
            "variance": variance,
            "low_trust_ratio": low_ratio,
        }

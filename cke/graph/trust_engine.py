"""Trust scoring utilities for assertions."""

from __future__ import annotations

import time
from pathlib import Path

from cke.graph.assertion import Assertion
from cke.trust.calibration import TrustCalibrationConfig, TrustCalibrator


class TrustEngine:
    """Compute trust scores from source quality, evidence, and recency."""

    def __init__(
        self,
        tau: float = 60.0 * 60.0 * 24.0 * 365.0,
        source_weights: dict[str, float] | None = None,
        calibrator: TrustCalibrator | None = None,
        config_path: str | Path | None = "configs/trust_config.yaml",
    ) -> None:
        self.source_weights = source_weights or {
            "wikipedia": 1.0,
            "paper": 1.2,
            "docs": 1.1,
            "unknown": 0.7,
        }
        calibration = self._load_config(config_path)
        if tau is not None:
            calibration.tau = tau
        self.calibrator = calibrator or TrustCalibrator(config=calibration)

    @staticmethod
    def _load_config(config_path: str | Path | None) -> TrustCalibrationConfig:
        """Load calibration config from YAML-like key:value file."""
        cfg = TrustCalibrationConfig()
        if config_path is None:
            return cfg
        path = Path(config_path)
        if not path.exists():
            return cfg
        data: dict[str, float] = {}
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                data[key.strip()] = float(value.strip())
            except ValueError:
                continue
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
        """Compute trust score with calibrated weighting and time decay."""
        source_weight = self.source_weights.get(
            assertion.source, self.source_weights["unknown"]
        )
        trust_0 = self.calibrator.calibrate(
            source_weight=source_weight,
            evidence_count=assertion.evidence_count,
            extractor_confidence=assertion.extractor_confidence,
        )
        now_ts = float(time.time()) if now is None else float(now)
        trust = self.calibrator.apply_decay(trust_0, now=now_ts, observed_ts=assertion.timestamp)
        assertion.trust_score = trust
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

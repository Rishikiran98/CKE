"""Token and cost tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass

from cke.diagnostics import DegradationMixin


@dataclass
class TokenTracker(DegradationMixin):
    tokens_prompt: int = 0
    tokens_completion: int = 0
    cost_per_1k_prompt: float = 0.0
    cost_per_1k_completion: float = 0.0

    def add_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.tokens_prompt += max(0, int(prompt_tokens))
        self.tokens_completion += max(0, int(completion_tokens))

    @property
    def total_tokens(self) -> int:
        return self.tokens_prompt + self.tokens_completion

    @property
    def cost_estimate(self) -> float:
        prompt_cost = (self.tokens_prompt / 1000.0) * self.cost_per_1k_prompt
        completion_cost = (
            self.tokens_completion / 1000.0
        ) * self.cost_per_1k_completion
        return prompt_cost + completion_cost

    @property
    def has_pricing(self) -> bool:
        """True when a rate was configured, so cost_estimate means something."""
        return bool(self.cost_per_1k_prompt or self.cost_per_1k_completion)

    def to_dict(self) -> dict[str, float | int | bool | None]:
        """Serialise usage.

        ``cost_estimate`` used to be emitted as 0.0 whenever no rate had been
        configured, which no caller in this package does, so every report
        carried a cost of zero as though it had been computed. It is None when
        there is no pricing to compute it from.
        """
        if not self.has_pricing:
            self._degrade(
                "no per-1k token prices are configured, so no cost can be "
                "computed. cost_estimate is reported as null rather than 0.0",
            )

        return {
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "total_tokens": self.total_tokens,
            "cost_estimate": self.cost_estimate if self.has_pricing else None,
            "cost_estimate_is_computed": self.has_pricing,
        }

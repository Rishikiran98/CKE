"""Token and cost tracking utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenTracker:
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
        completion_cost = (self.tokens_completion / 1000.0) * self.cost_per_1k_completion
        return prompt_cost + completion_cost

    def to_dict(self) -> dict[str, float | int]:
        return {
            "tokens_prompt": self.tokens_prompt,
            "tokens_completion": self.tokens_completion,
            "total_tokens": self.total_tokens,
            "cost_estimate": self.cost_estimate,
        }

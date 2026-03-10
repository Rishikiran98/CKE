"""LLM-backed reasoner with deterministic template fallback."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, List
from urllib import request

from cke.models import Statement
from cke.reasoning.reasoner import TemplateReasoner

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


@dataclass(slots=True)
class LLMReasonerConfig:
    """Configuration for OpenAI-compatible reasoning requests."""

    endpoint: str = "https://api.openai.com/v1/chat/completions"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    timeout_s: float = 20.0


class LLMReasoner:
    """Reasoner that answers using graph context with robust fallback."""

    def __init__(
        self,
        config: LLMReasonerConfig | None = None,
        fallback: TemplateReasoner | None = None,
    ) -> None:
        self.config = config or LLMReasonerConfig(
            endpoint=os.getenv(
                "CKE_LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions"
            ),
            model=os.getenv("CKE_LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("CKE_LLM_API_KEY"),
        )
        self.fallback = fallback or TemplateReasoner()

    def answer(self, question: str, context: List[Statement]) -> str:
        """Answer a question grounded in graph statements."""
        if not self.config.api_key:
            return self.fallback.answer(question, context)

        if not context:
            return self.fallback.answer(question, context)

        try:
            payload = self._call_model(question, context)
            answer_text = self._parse_answer(payload)
            return answer_text or self.fallback.answer(question, context)
        except Exception:
            return self.fallback.answer(question, context)

    def format_reasoning_path(self, context: List[Statement]) -> str:
        """Expose same interface as TemplateReasoner."""
        return self.fallback.format_reasoning_path(context)

    def _build_prompt(self, question: str, context: List[Statement]) -> str:
        evidence_lines = [
            f"- {st.as_text()} (confidence={st.confidence:.2f})" for st in context
        ]
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "- (none)"
        return (
            "You are a grounded QA assistant for a knowledge graph.\n"
            "Answer ONLY using the provided graph context evidence.\n"
            "If evidence is insufficient, explicitly say that the graph context is insufficient.\n"
            'Return JSON only with schema: {"answer": "...", "used_evidence": ["..."]}.\n\n'
            f"Question: {question}\n"
            "Graph context evidence:\n"
            f"{evidence_text}\n"
        )

    def _call_model(self, question: str, context: List[Statement]) -> dict[str, Any]:
        if OpenAI is not None:
            try:
                client = OpenAI(api_key=self.config.api_key)
                response = client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {
                            "role": "user",
                            "content": self._build_prompt(question, context),
                        }
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content
                return {"choices": [{"message": {"content": content}}]}
            except Exception:
                pass

        body = {
            "model": self.config.model,
            "messages": [
                {"role": "user", "content": self._build_prompt(question, context)}
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        req = request.Request(
            self.config.endpoint, data=payload, headers=headers, method="POST"
        )
        with request.urlopen(req, timeout=self.config.timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def _parse_answer(self, payload: dict[str, Any]) -> str:
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        answer = parsed.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()
        return ""

"""LLM-backed assertion extractor with retry and JSON validation."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from cke.extractor.extractor import BaseExtractor, RuleBasedExtractor
from cke.models import Statement
from cke.observability.token_tracker import TokenTracker

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class AssertionPayload(BaseModel):
    """Structured assertion payload returned by the LLM."""

    subject: str
    relation: str
    object: str
    confidence: float = 1.0
    evidence: list[dict[str, Any]] = Field(default_factory=list)


@dataclass(slots=True)
class LLMConfig:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    max_retries: int = 2
    retry_delay_s: float = 0.3


class LLMExtractor(BaseExtractor):
    """Extract structured assertions from text using an LLM."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        fallback: BaseExtractor | None = None,
        token_tracker: TokenTracker | None = None,
    ) -> None:
        self.config = config or LLMConfig(api_key=os.getenv("CKE_LLM_API_KEY"))
        self.fallback = fallback or RuleBasedExtractor()
        self.token_tracker = token_tracker or TokenTracker()
        self.client = (
            OpenAI(api_key=self.config.api_key)
            if (OpenAI and self.config.api_key)
            else None
        )

    def extract(self, text: str) -> list[Statement]:
        if self.client is None:
            return self.fallback.extract(text)

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                payload = self._call_llm(text)
                statements = self._parse_response(payload, source_text=text)
                return statements or self.fallback.extract(text)
            except Exception as exc:  # pragma: no cover - network/runtime variability
                last_error = exc
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay_s)
        if last_error:
            return self.fallback.extract(text)
        return []

    def _call_llm(self, text: str) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("LLM client is not configured.")

        prompt = (
            "Extract factual assertions as JSON array with keys "
            "subject, relation, object, confidence, evidence. "
            "Each evidence item must include chunk_id, span_start, span_end, "
            "text, extractor_confidence, source_weight. "
            "Span offsets are character offsets over the provided text. "
            "Return only JSON."
        )
        completion = self.client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        usage = getattr(completion, "usage", None)
        if usage is not None:
            self.token_tracker.add_usage(
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            )
        content = completion.choices[0].message.content or "[]"
        return {"choices": [{"message": {"content": content}}]}

    def _parse_response(
        self,
        payload: dict[str, Any],
        source_text: str | None = None,
    ) -> list[Statement]:
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "triples" in parsed:
            parsed = parsed["triples"]
        if not isinstance(parsed, list):
            raise ValueError("LLM response must be a list of assertions.")

        statements: list[Statement] = []
        for item in parsed:
            try:
                assertion = AssertionPayload(**item)
            except ValidationError:
                continue
            subject = self._clean_token(assertion.subject)
            relation = self._clean_relation(assertion.relation)
            obj = self._clean_token(assertion.object)
            evidence = self._validated_evidence(assertion.evidence, source_text)
            if source_text is not None and not evidence:
                continue
            if subject and relation and obj:
                statements.append(
                    Statement(
                        subject=subject,
                        relation=relation,
                        object=obj,
                        confidence=float(assertion.confidence),
                        context={"evidence": evidence},
                    )
                )
        return statements

    def _validated_evidence(
        self,
        evidence_items: list[dict[str, Any]],
        source_text: str | None,
    ) -> list[dict[str, Any]]:
        """Validate span-level evidence and normalize metadata."""
        if source_text is None:
            return list(evidence_items)

        validated: list[dict[str, Any]] = []
        for item in evidence_items:
            try:
                start = int(item["span_start"])
                end = int(item["span_end"])
                text = str(item["text"])
            except (KeyError, TypeError, ValueError):
                continue

            if start < 0 or end <= start or end > len(source_text):
                continue
            if source_text[start:end] != text:
                continue
            validated.append(
                {
                    "chunk_id": str(item.get("chunk_id", "chunk-0")),
                    "span_start": start,
                    "span_end": end,
                    "text": text,
                    "extractor_confidence": float(
                        item.get("extractor_confidence", 1.0)
                    ),
                    "source_weight": float(item.get("source_weight", 1.0)),
                }
            )
        return validated

    @staticmethod
    def _clean_token(token: str) -> str:
        return re.sub(r"\s+", " ", token.strip(" .,:;\n\t"))

    @staticmethod
    def _clean_relation(relation: str) -> str:
        rel = re.sub(r"\s+", "_", relation.strip().lower())
        return re.sub(r"[^a-z0-9_]", "", rel)

"""LLM-backed assertion extractor with retry and JSON validation."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from cke.extractor.extractor import BaseExtractor, RuleBasedExtractor
from cke.models import Statement

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


class AssertionPayload(BaseModel):
    subject: str
    relation: str
    object: str
    confidence: float = 1.0


@dataclass(slots=True)
class LLMConfig:
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    max_retries: int = 2
    retry_delay_s: float = 0.3


class LLMExtractor(BaseExtractor):
    """Extract structured assertions from text using an LLM."""

    def __init__(
        self, config: LLMConfig | None = None, fallback: BaseExtractor | None = None
    ) -> None:
        self.config = config or LLMConfig(api_key=os.getenv("CKE_LLM_API_KEY"))
        self.fallback = fallback or RuleBasedExtractor()
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
                statements = self._parse_response(payload)
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
            "subject, relation, object, confidence. Return only JSON."
        )
        completion = self.client.chat.completions.create(
            model=self.config.model,
            temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
        )
        content = completion.choices[0].message.content or "[]"
        return {"choices": [{"message": {"content": content}}]}

    def _parse_response(self, payload: dict[str, Any]) -> list[Statement]:
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
            if subject and relation and obj:
                statements.append(
                    Statement(
                        subject=subject,
                        relation=relation,
                        object=obj,
                        confidence=float(assertion.confidence),
                    )
                )
        return statements

    @staticmethod
    def _clean_token(token: str) -> str:
        return re.sub(r"\s+", " ", token.strip(" .,:;\n\t"))

    @staticmethod
    def _clean_relation(relation: str) -> str:
        rel = re.sub(r"\s+", "_", relation.strip().lower())
        return re.sub(r"[^a-z0-9_]", "", rel)

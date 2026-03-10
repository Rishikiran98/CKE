"""LLM-backed semantic extractor with rule-based fallback.

This module is intentionally lightweight for local prototype usage:
- If LLM credentials/config are available, it calls a compatible chat endpoint.
- If unavailable or the call fails, it falls back to the rule extractor.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, List
from urllib import error, request

from cke.extractor.extractor import BaseExtractor, RuleBasedExtractor
from cke.models import Statement


@dataclass(slots=True)
class LLMConfig:
    """Configuration for a generic OpenAI-compatible chat endpoint."""

    endpoint: str = "https://api.openai.com/v1/chat/completions"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    timeout_s: float = 20.0


class LLMExtractor(BaseExtractor):
    """Extract triples via LLM with rule-based fallback."""

    def __init__(self, config: LLMConfig | None = None, fallback: BaseExtractor | None = None) -> None:
        self.config = config or LLMConfig(
            endpoint=os.getenv("CKE_LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions"),
            model=os.getenv("CKE_LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("CKE_LLM_API_KEY"),
        )
        self.fallback = fallback or RuleBasedExtractor()

    def extract(self, text: str) -> List[Statement]:
        if not self.config.api_key:
            return self.fallback.extract(text)

        try:
            response_payload = self._call_llm(text)
            statements = self._parse_response(response_payload)
            return statements or self.fallback.extract(text)
        except Exception:
            return self.fallback.extract(text)

    def _call_llm(self, text: str) -> dict[str, Any]:
        prompt = (
            "Extract subject-relation-object triples from the input text. "
            "Return ONLY JSON with this schema: "
            '{"triples":[{"subject":"...","relation":"...","object":"..."}]}. '
            "Use snake_case relations where possible."
        )
        body = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        req = request.Request(self.config.endpoint, data=payload, headers=headers, method="POST")
        with request.urlopen(req, timeout=self.config.timeout_s) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def _parse_response(self, payload: dict[str, Any]) -> List[Statement]:
        # OpenAI-compatible response payload
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        triples = parsed.get("triples", [])

        statements: list[Statement] = []
        for triple in triples:
            subject = self._clean_token(str(triple.get("subject", "")))
            relation = self._clean_relation(str(triple.get("relation", "")))
            object_ = self._clean_token(str(triple.get("object", "")))
            if subject and relation and object_:
                statements.append(Statement(subject, relation, object_))
        return statements

    @staticmethod
    def _clean_token(token: str) -> str:
        return re.sub(r"\s+", " ", token.strip(" .,:;\n\t"))

    @staticmethod
    def _clean_relation(relation: str) -> str:
        rel = re.sub(r"\s+", "_", relation.strip().lower())
        return re.sub(r"[^a-z0-9_]", "", rel)

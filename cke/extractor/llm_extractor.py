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
from cke.reasoning.reasoning_trace import ReasoningTrace, ReasoningTraceLogger

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


_VALID_QUALIFIER_KEYS = {"temporal", "condition", "scope", "modality"}
_TEMPORAL_KEYS = {"start", "end", "observed_at"}
_CONDITION_KEYS = {"environment", "when", "unless"}
_SCOPE_KEYS = {"version", "jurisdiction"}
_MODALITY_VALUES = {"typical", "possible", "rare", "deprecated", "unknown"}


def validate_qualifiers(qualifiers: dict[str, Any]) -> bool:
    """Return True if *qualifiers* conforms to the document schema."""
    if not qualifiers:
        return True
    if not isinstance(qualifiers, dict):
        return False
    for key, value in qualifiers.items():
        if key not in _VALID_QUALIFIER_KEYS:
            return False
        if key == "temporal":
            if not isinstance(value, dict):
                return False
            if not set(value.keys()).issubset(_TEMPORAL_KEYS):
                return False
        elif key == "condition":
            if not isinstance(value, dict):
                return False
            if not set(value.keys()).issubset(_CONDITION_KEYS):
                return False
        elif key == "scope":
            if not isinstance(value, dict):
                return False
            if not set(value.keys()).issubset(_SCOPE_KEYS):
                return False
        elif key == "modality":
            if not isinstance(value, str) or value not in _MODALITY_VALUES:
                return False
    return True


class AssertionPayload(BaseModel):
    """Structured assertion payload returned by the LLM."""

    subject: str
    relation: str
    object: str | dict[str, Any]
    confidence: float = 1.0
    extractor_confidence: float = 1.0
    qualifiers: dict[str, Any] = Field(default_factory=dict)
    evidence: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def object_value(self) -> str:
        """Return the plain string value of object, handling typed dicts."""
        if isinstance(self.object, dict):
            return str(self.object.get("value", ""))
        return str(self.object)


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
        self.trace_logger = ReasoningTraceLogger()

    def extract(self, text: str) -> list[Statement]:
        if self.client is None:
            return self.fallback.extract(text)

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                payload = self._call_llm(text)
                statements = self._parse_response(payload, source_text=text)
                if statements:
                    self.trace_logger.write(
                        ReasoningTrace(
                            query=text,
                            entities=sorted({s.subject for s in statements}),
                            retrieved_facts=[
                                {
                                    "subject": st.subject,
                                    "relation": st.relation,
                                    "object": st.object,
                                    "confidence": st.confidence,
                                }
                                for st in statements
                            ],
                            confidence_score=sum(st.confidence for st in statements)
                            / max(1, len(statements)),
                            final_answer="extraction_complete",
                            operators_used=["llm_extraction"],
                        ),
                        stage="llm_extractor",
                    )
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
            "Extract factual assertions from the text as a JSON array. "
            "Each assertion object must follow this schema:\n"
            '{"subject": "string", '
            '"relation": "string", '
            '"object": {"type": "entity|literal", "value": "string"}, '
            '"qualifiers": {'
            '"temporal": {"start": null, "end": null, "observed_at": null}, '
            '"condition": {"environment": null, "when": null, "unless": null}, '
            '"scope": {"version": null, "jurisdiction": null}, '
            '"modality": "typical|possible|rare|deprecated|unknown"}, '
            '"extractor_confidence": 0.0, '
            '"confidence": 0.0, '
            '"evidence": [{"chunk_id": "string", "span_start": 0, '
            '"span_end": 0, "text": "string", '
            '"extractor_confidence": 0.0, "source_weight": 0.0}]}\n'
            "Rules:\n"
            "- Only include qualifier keys when the text explicitly states them; "
            "omit null-valued qualifier sub-keys.\n"
            "- object.type is 'entity' for named entities, 'literal' for values.\n"
            "- Span offsets are character offsets over the provided text.\n"
            "- extractor_confidence (0-1) reflects how certain you are.\n"
            "- Return only JSON, no explanation."
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
            obj = self._clean_token(assertion.object_value)
            if not validate_qualifiers(assertion.qualifiers):
                continue
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
                        qualifiers=dict(assertion.qualifiers),
                        context={
                            "evidence": evidence,
                            "extractor_confidence": float(
                                assertion.extractor_confidence
                            ),
                        },
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

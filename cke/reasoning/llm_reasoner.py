"""LLM-backed reasoner with deterministic template fallback.

Without an API key this class does not reason with an LLM at all: it returns a
string-template answer that is indistinguishable, to a caller, from a generated
one. The substitution is therefore declared rather than made silently.
"""

from __future__ import annotations

import json
import logging
import os
import re
import string
from dataclasses import dataclass
from typing import Any, Iterable, List
from urllib import request

from cke.diagnostics import DegradationMixin
from cke.models import Statement
from cke.reasoning.path_reasoner import PathReasoner
from cke.reasoning.reasoner import TemplateReasoner

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime dependency
    OpenAI = None


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMReasonerConfig:
    """Configuration for OpenAI-compatible reasoning requests."""

    endpoint: str = "https://api.openai.com/v1/chat/completions"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    timeout_s: float = 20.0


class LLMReasoner(DegradationMixin):
    """Reasoner that answers using graph context with robust fallback.

    Args:
        config: endpoint, model and API-key configuration.
        fallback: reasoner used when no LLM is reachable.
        strict: when True, raise rather than answer from the template
            fallback. Every evaluation and benchmark path must pass
            ``strict=True``.
    """

    def __init__(
        self,
        config: LLMReasonerConfig | None = None,
        fallback: PathReasoner | TemplateReasoner | None = None,
        strict: bool = False,
    ) -> None:
        self._init_degradation(strict)
        self.config = config or LLMReasonerConfig(
            endpoint=os.getenv(
                "CKE_LLM_ENDPOINT",
                "https://api.openai.com/v1/chat/completions",
            ),
            model=os.getenv("CKE_LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("CKE_LLM_API_KEY"),
        )
        self.fallback = fallback or PathReasoner()
        self.last_evidence_ids: list[str] = []
        self.last_trace: str = ""

        if not self.config.api_key:
            self._degrade(
                "no API key is configured; set CKE_LLM_API_KEY or pass "
                "LLMReasonerConfig(api_key=...), so no LLM is in the loop and "
                f"answers come from {type(self.fallback).__name__} string "
                "templates. Any answer quality measured in this state "
                "describes that fallback, not an LLM"
            )

    def answer(self, question: str, context: Iterable[Any]) -> str:
        """Answer a question grounded in graph statements."""
        normalized_context = self._normalize_context(context)
        selected_context = self._select_context(question, normalized_context)

        if not self.config.api_key:
            return self.fallback.answer(question, selected_context)

        if not selected_context:
            return self.fallback.answer(question, selected_context)

        try:
            payload = self._call_model(question, selected_context)
            answer_text = self._parse_answer(payload)
        except Exception as exc:  # noqa: BLE001 - network/runtime variability
            self._degrade(
                f"the LLM call failed ({type(exc).__name__}: {exc}), so this "
                f"answer came from {type(self.fallback).__name__} instead"
            )
            return self.fallback.answer(question, selected_context)

        if not answer_text:
            self._degrade(
                "the LLM returned no usable answer, so this answer came from "
                f"{type(self.fallback).__name__} instead"
            )
            return self.fallback.answer(question, selected_context)
        return answer_text

    def format_reasoning_path(self, context: List[Statement]) -> str:
        """Expose same interface as TemplateReasoner."""
        return self.fallback.format_reasoning_path(context)

    def _build_prompt(self, question: str, context: List[Statement]) -> str:
        evidence_lines = [
            f"[E{i}] {st.as_text()} (confidence={st.confidence:.2f})"
            for i, st in enumerate(context, start=1)
        ]
        evidence_text = "\n".join(evidence_lines) if evidence_lines else "- (none)"
        return (
            "You are a grounded QA assistant for a knowledge graph.\n"
            "Task: answer the QUESTION, not just summarize evidence.\n"
            "Use only the provided graph evidence.\n"
            "If evidence is insufficient, say: 'Insufficient graph evidence.'\n"
            "Prefer evidence that directly mentions entities/relations "
            "in the question.\n"
            "Return JSON only with schema: "
            '{"answer": "...", '
            '"evidence_ids": ["E1", "E2"], '
            '"trace": "brief reasoning chain explaining how evidence '
            'supports the answer"}.\n'
            "evidence_ids must reference the [E#] labels from the context.\n\n"
            f"QUESTION: {question}\n"
            "Graph context evidence:\n"
            f"{evidence_text}\n"
        )

    def _normalize_context(self, context: Iterable[Any]) -> list[Statement]:
        normalized: list[Statement] = []
        for item in context:
            if isinstance(item, Statement):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                subject = item.get("subject")
                relation = item.get("relation")
                object_ = item.get("object")
                if not all(
                    isinstance(v, str) and v for v in (subject, relation, object_)
                ):
                    continue
                confidence = item.get("trust_score", item.get("trust", 1.0))
                try:
                    confidence_value = float(confidence)
                except (TypeError, ValueError):
                    confidence_value = 1.0
                normalized.append(
                    Statement(
                        subject=subject,
                        relation=relation,
                        object=object_,
                        confidence=confidence_value,
                    )
                )
        return normalized

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
            except Exception as exc:  # noqa: BLE001 - SDK raises varied errors
                # The SDK path failed. Retrying the same request over the raw
                # HTTP path below is only worth doing for a transport problem,
                # and it must not hide an auth or model error, so say what
                # happened before falling through.
                logger.warning(
                    "OpenAI SDK call failed (%s: %s); retrying over plain HTTP",
                    type(exc).__name__,
                    exc,
                )

        body = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": self._build_prompt(question, context),
                }
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        if not self.config.endpoint.startswith(("http://", "https://")):
            raise ValueError(f"Invalid endpoint scheme: {self.config.endpoint}")
        req = request.Request(
            self.config.endpoint, data=payload, headers=headers, method="POST"
        )
        with request.urlopen(req, timeout=self.config.timeout_s) as resp:  # nosec B310
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    def _select_context(
        self, question: str, context: List[Statement], limit: int = 10
    ) -> list[Statement]:
        if len(context) <= limit:
            return context

        question_tokens = set(re.findall(r"\w+", question.lower()))

        def evidence_score(statement: Statement) -> tuple[int, float]:
            statement_text = (
                f"{statement.subject} {statement.relation} {statement.object}"
            )
            evidence_tokens = set(re.findall(r"\w+", statement_text.lower()))
            overlap = len(question_tokens.intersection(evidence_tokens))
            return overlap, statement.confidence

        ranked = sorted(context, key=evidence_score, reverse=True)
        return ranked[:limit]

    def _parse_answer(self, payload: dict[str, Any]) -> str:
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)

        # Capture evidence_ids and trace from the grounded answer format.
        evidence_ids = parsed.get("evidence_ids")
        if isinstance(evidence_ids, list):
            self.last_evidence_ids = [str(e) for e in evidence_ids]
        else:
            # Backward compat: fall back to used_evidence if present.
            used = parsed.get("used_evidence")
            self.last_evidence_ids = (
                [str(e) for e in used] if isinstance(used, list) else []
            )
        trace = parsed.get("trace")
        self.last_trace = str(trace) if trace else ""

        answer = parsed.get("answer")
        if isinstance(answer, str) and answer.strip():
            return self._normalize_answer(answer)
        return ""

    def _normalize_answer(self, answer: str) -> str:
        cleaned = answer.strip()
        cleaned = cleaned.strip("\"'`“”‘’")
        cleaned = re.sub(r"^the answer is\s+", "", cleaned, flags=re.IGNORECASE)
        first_segment = re.split(r"(?<=[.!?])\s+|\n", cleaned, maxsplit=1)[0]
        normalized = first_segment.strip().strip("\"'`“”‘’")
        normalized = normalized.strip(string.punctuation + "“”‘’")
        return normalized.strip()

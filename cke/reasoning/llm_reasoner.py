"""LLM-backed reasoner with deterministic template fallback."""

from __future__ import annotations

import json
import os
import re
import string
from dataclasses import dataclass
from typing import Any, Iterable, List
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
                "CKE_LLM_ENDPOINT",
                "https://api.openai.com/v1/chat/completions",
            ),
            model=os.getenv("CKE_LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("CKE_LLM_API_KEY"),
        )
        self.fallback = fallback or TemplateReasoner()

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
            return answer_text or self.fallback.answer(question, selected_context)
        except Exception:
            return self.fallback.answer(question, selected_context)

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
            '{"answer": "...", "used_evidence": ["..."]}.\n\n'
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
            except Exception:  # nosec B110
                pass

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
            evidence_tokens = set(
                re.findall(
                    r"\w+",
                    f"{statement.subject} {statement.relation} {statement.object}".lower(),
                )
            )
            overlap = len(question_tokens.intersection(evidence_tokens))
            return overlap, statement.confidence

        ranked = sorted(context, key=evidence_score, reverse=True)
        return ranked[:limit]

    def _parse_answer(self, payload: dict[str, Any]) -> str:
        content = payload["choices"][0]["message"]["content"]
        parsed = json.loads(content)
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

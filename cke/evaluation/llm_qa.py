"""Answer a question from retrieved context with a language model.

This is the answerer the comparison has been missing. Every arm so far has
been read by :class:`~cke.evaluation.span_qa.SpanExtractiveQA`, a lexical
span baseline with no learned components, which bounds both arms and cannot
tell a compressed context from a poor one: a matcher scanning 1250 tokens of
prose for an overlapping span has far more surface to hit than one scanning
36 tokens of triples. Whether CKE's compression costs accuracy is a question
about what reads the context, and only a model can answer it.

Symmetry
--------
One instance, one prompt, one decoding setting, for every arm. The model sees
a question and a context string and cannot tell which retrieval produced the
context. The context is passed exactly as each arm assembles it — prose for
the dense arm, ``subject relation object`` lines for the graph arm — so the
comparison measures retrieval, not rendering.

Two backends
------------
``local`` runs an instruction-tuned sequence-to-sequence model through
``transformers`` on this machine; ``api`` posts to an OpenAI-compatible
chat-completions endpoint using the same ``CKE_LLM_*`` variables as
:class:`~cke.reasoning.llm_reasoner.LLMReasoner`. Neither has a fallback. A
backend that cannot be brought up declares the degradation and, under strict,
refuses: an evaluation with no model in the loop is the lexical baseline's,
and must say so rather than produce a number.

Truncation is a measurement, not a nuisance. A model has a context window, a
dense arm's prose can exceed it, and dropping the tail of the context changes
what the model was shown. It is counted per call, per arm, and reported.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from urllib import request

from cke.diagnostics import DegradationMixin, record_loaded_model

__all__ = ["LLMAnswerer", "PROMPT"]

#: The one prompt. Written once, on the shape of an extractive QA instruction,
#: and not revised against any benchmark: revising it until a score moved
#: would make the score describe the prompt.
PROMPT = (
    "Answer the question using only the context. Reply with the shortest "
    "span of text that answers it, or 'unknown' if the context does not "
    "say.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
)

#: Longest answer decoded. Gold answers are short; a long generation would
#: reintroduce the sentence-prefix problem the span answerer was built to end.
_MAX_NEW_TOKENS = 16

_DEFAULT_LOCAL_MODEL = "google/flan-t5-base"


@dataclass
class TruncationLog:
    """How often the context did not fit the model's window."""

    calls: int = 0
    truncated: int = 0
    dropped_tokens: list[int] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.truncated / self.calls if self.calls else 0.0


class LLMAnswerer(DegradationMixin):
    """Answer with a language model, or refuse."""

    def __init__(
        self,
        backend: str = "local",
        model: str | None = None,
        strict: bool = False,
        api_key: str | None = None,
        endpoint: str | None = None,
        timeout_s: float = 30.0,
        max_input_tokens: int | None = None,
    ) -> None:
        """
        ``max_input_tokens`` is the longest prompt handed to a local model.
        None means the window the model reports for itself, which is what it
        was trained with. A wider value is permitted because T5-family
        attention is relative and runs on longer inputs, but those inputs are
        outside what the model saw, so a run using one is labelled with it.
        The choice changes which arm truncation hits — a dense arm's prose
        exceeds 512 tokens on most questions, a graph arm's triples on none —
        and is therefore reported, never picked to favour an arm.
        """
        self._init_degradation(strict)
        self._requested_window = max_input_tokens
        if backend not in ("local", "api"):
            raise ValueError(f"backend must be 'local' or 'api', not {backend!r}")
        self.backend = backend
        self.truncation = TruncationLog()
        self._tokenizer = None
        self._model = None
        self._window = 0

        if backend == "local":
            self.model_name = model or _DEFAULT_LOCAL_MODEL
            self._load_local()
        else:
            self.model_name = model or os.getenv("CKE_LLM_MODEL", "gpt-4o-mini")
            self.endpoint = endpoint or os.getenv(
                "CKE_LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions"
            )
            self.api_key = api_key or os.getenv("CKE_LLM_API_KEY")
            self.timeout_s = timeout_s
            if not self.api_key:
                self._degrade(
                    "no API key is configured; set CKE_LLM_API_KEY or pass "
                    "api_key=..., so no model can be called and no answer can "
                    "be produced"
                )

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    def _load_local(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:
            self._degrade(
                f"transformers or torch is not installed ({exc}), so no local "
                f"model can be loaded. Install them with `pip install "
                f"transformers torch`"
            )
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        except Exception as exc:  # noqa: BLE001 - hub and load errors vary
            self._degrade(
                f"the local model {self.model_name!r} could not be loaded "
                f"({type(exc).__name__}: {exc}), so no answer can be produced"
            )
            self._tokenizer = self._model = None
            return
        self._model.eval()
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        # The window the model was trained with. flan-t5 reports 512; the
        # attention is relative so longer inputs run, but they are outside
        # what the model saw and are counted as truncation against it.
        native = int(getattr(self._tokenizer, "model_max_length", 512))
        self._window = self._requested_window or native
        record_loaded_model("LLMAnswerer", self.model_name, self.model_name)

    @property
    def available(self) -> bool:
        if self.backend == "local":
            return self._model is not None
        return bool(self.api_key)

    @property
    def description(self) -> str:
        """How the answers were produced, for printing beside a figure."""
        if not self.available:
            return f"NO MODEL ({self.degraded_reason})"
        where = (
            f"local, transformers, {self._window}-token window"
            if self.backend == "local"
            else self.endpoint
        )
        return (
            f"LLMAnswerer {self.model_name} ({where}; greedy, "
            f"{_MAX_NEW_TOKENS} tokens)"
        )

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------

    def answer(self, question: str, context: str) -> str:
        if not self.available:
            raise RuntimeError(
                f"LLMAnswerer has no model to answer with: {self.degraded_reason}"
            )
        prompt = PROMPT.format(context=context.strip(), question=question.strip())
        if self.backend == "local":
            return self._answer_local(prompt)
        return self._answer_api(prompt)

    def _answer_local(self, prompt: str) -> str:
        import torch

        ids = self._tokenizer(prompt, return_tensors="pt")
        length = int(ids["input_ids"].shape[-1])
        self.truncation.calls += 1
        if length > self._window:
            self.truncation.truncated += 1
            self.truncation.dropped_tokens.append(length - self._window)
            ids = self._tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=self._window
            )
        with torch.no_grad():
            out = self._model.generate(
                **ids, max_new_tokens=_MAX_NEW_TOKENS, do_sample=False
            )
        return self._tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def _answer_api(self, prompt: str) -> str:
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": _MAX_NEW_TOKENS,
        }
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError(f"Invalid endpoint scheme: {self.endpoint}")
        req = request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        self.truncation.calls += 1
        with request.urlopen(req, timeout=self.timeout_s) as resp:  # nosec B310
            payload = json.loads(resp.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"]).strip()

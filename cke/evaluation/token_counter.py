"""Count tokens with a real tokenizer.

This replaces a counter that multiplied a whitespace word count by 1.3. Every
"prompt tokens" figure the benchmark reported came from that multiplier, so no
figure was a measurement, and the ratio between two arms followed partly from
the units each arm counted in rather than from retrieval.

What a token count means
------------------------
Nothing, on its own. A token is defined by a particular tokenizer, so a count
is only meaningful once the encoding is named. This class names it: every
counter carries a :attr:`description`, and callers are expected to print it
beside any figure they report. Two counts are comparable only when they were
produced by the same encoding.

No language model is prompted anywhere in this repository, so these counts do
not describe what any model would consume. They measure the size of a context
under a named, standard encoding, which is what makes two retrieval strategies
comparable to each other.
"""

from __future__ import annotations

from cke.diagnostics import DegradationMixin, record_loaded_model

__all__ = ["TokenCounter"]

#: Fallback multiplier, used only on the declared degraded path. This is the
#: constant the whole class exists to replace; it is named here so that a
#: reader of a degraded run can see exactly what produced the number.
_WORDS_TO_TOKENS = 1.3


class TokenCounter(DegradationMixin):
    """Count tokens in a string under a named encoding.

    Construct with ``strict=True`` — as every evaluation entry point does — and
    a missing tokenizer stops the run instead of quietly restoring the word
    count estimate.
    """

    #: The encoding counted in by default. cl100k_base is the encoding of the
    #: GPT-3.5 and GPT-4 families and is the most widely reported one, which
    #: makes a figure produced with it comparable to published figures.
    DEFAULT_ENCODING = "cl100k_base"

    def __init__(self, encoding: str = DEFAULT_ENCODING, strict: bool = False) -> None:
        self._init_degradation(strict)
        self._encoding_name = encoding
        self._encoding = self._load_encoding(encoding)

    def _load_encoding(self, encoding: str):
        """Load the tokenizer, or declare the degradation and return None."""
        try:
            import tiktoken
        except ImportError as exc:
            self._degrade(
                f"tiktoken is not installed ({exc}), so tokens are estimated as "
                f"word count x {_WORDS_TO_TOKENS}. That is a multiplier, not a "
                f"tokenizer, and a figure produced by it is not a measurement. "
                f"Install it with `pip install tiktoken`"
            )
            return None

        try:
            loaded = tiktoken.get_encoding(encoding)
        except Exception as exc:  # noqa: BLE001 - tiktoken raises varied errors
            # Broad by necessity: the first use of an encoding fetches its BPE
            # table over the network, so this covers ValueError for an unknown
            # name and any transport error underneath. Every one of them is
            # reported in the degradation reason rather than discarded.
            self._degrade(
                f"the {encoding!r} encoding could not be loaded ({exc}), so "
                f"tokens are estimated as word count x {_WORDS_TO_TOKENS}. That "
                f"is a multiplier, not a tokenizer, and a figure produced by it "
                f"is not a measurement"
            )
            return None

        record_loaded_model("TokenCounter", encoding, loaded.name)
        return loaded

    @property
    def encoding_name(self) -> str:
        """The encoding asked for, whether or not it loaded."""
        return self._encoding_name

    @property
    def is_estimate(self) -> bool:
        """True when counts come from the multiplier rather than a tokenizer."""
        return self._encoding is None

    @property
    def description(self) -> str:
        """How the counts were produced, for printing beside a figure."""
        if self._encoding is None:
            return (
                f"ESTIMATE: word count x {_WORDS_TO_TOKENS}, not tokenizer "
                f"output ({self.degraded_reason})"
            )
        return f"tiktoken {self._encoding.name}"

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encoding is None:
            return max(1, int(len(text.split()) * _WORDS_TO_TOKENS))
        # disallowed_special=() encodes a literal such as "<|endoftext|>" as
        # ordinary characters. The default raises on it, which would abort a
        # run over arbitrary dataset text; and this counter is only ever asked
        # to measure text, never to build a prompt, so a control token here is
        # a string a document happened to contain and nothing more.
        return len(self._encoding.encode(text, disallowed_special=()))

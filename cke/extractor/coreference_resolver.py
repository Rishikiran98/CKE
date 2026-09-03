"""Coreference resolution for document-level extraction.

Without a spaCy model this falls back to a regular expression that rewrites
every pronoun to the most recently seen capitalised name. That is a different
algorithm with a different error profile, not a slightly worse version of the
same one, so the fallback is declared.
"""

from __future__ import annotations

import logging
import re

from cke.diagnostics import DegradationMixin

try:
    import spacy
except ImportError:  # pragma: no cover - optional runtime dependency
    spacy = None


logger = logging.getLogger(__name__)

_SPACY_MODELS = ("en_coreference_web_trf", "en_core_web_sm")


class CoreferenceResolver(DegradationMixin):
    """Resolve pronoun references to the latest salient named entity.

    Args:
        strict: when True, raise rather than fall back to the regex heuristic.
    """

    PRONOUNS = {"he", "she", "they", "it", "him", "her", "them", "his", "its", "their"}
    _PERSON_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
    _NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

    def __init__(self, strict: bool = False) -> None:
        self._init_degradation(strict)
        self._nlp = self._load_spacy()

    def resolve(self, document: str) -> str:
        if not document.strip():
            return document
        resolved = self._resolve_with_spacy(document)
        if resolved is not None:
            return resolved
        if self._nlp is not None:
            # A model is loaded but produced no entities for this document, so
            # this answer comes from the regex, not from the model. Without
            # this the loaded-model path silently used the fallback.
            self._degrade(
                "a spaCy model is loaded but found no entities in a document, "
                "so coreference for it fell back to the regular expression "
                "that rewrites every pronoun to the most recently seen "
                "capitalised name"
            )
        return self._resolve_heuristic(document)

    def _load_spacy(self):
        if spacy is None:
            self._degrade(
                "spacy is not installed, so coreference resolution falls back "
                "to a regular expression that rewrites every pronoun to the "
                "most recently seen capitalised name. Install it with "
                "`pip install spacy`"
            )
            return None

        failures: list[str] = []
        for model_name in _SPACY_MODELS:
            model = self._try_load_spacy_model(model_name, failures)
            if model is not None:
                return model

        self._degrade(
            "spacy is installed but none of its coreference models loaded "
            f"({'; '.join(failures)}), so coreference resolution falls back to "
            "a regular expression that rewrites every pronoun to the most "
            "recently seen capitalised name. Install one with "
            f"`python -m spacy download {_SPACY_MODELS[-1]}`"
        )
        return None

    @staticmethod
    def _try_load_spacy_model(model_name: str, failures: list[str] | None = None):
        try:
            return spacy.load(model_name)
        except Exception as exc:  # noqa: BLE001 - spacy raises varied load errors
            # Model not present or incompatible in current runtime.
            if failures is not None:
                failures.append(f"{model_name}: {type(exc).__name__}: {exc}")
            logger.debug("spaCy model %s did not load: %s", model_name, exc)
            return None

    def _resolve_with_spacy(self, document: str) -> str | None:
        if self._nlp is None:
            return None
        doc = self._nlp(document)
        if not doc.ents:
            return None
        antecedent = None
        output_tokens: list[str] = []
        ent_starts = {ent.start: ent for ent in doc.ents}
        consumed_ent_tokens: set[int] = set()
        for token in doc:
            entity = ent_starts.get(token.i)
            if entity is not None:
                antecedent = entity.text
                output_tokens.append(entity.text + entity[-1].whitespace_)
                consumed_ent_tokens.update(range(entity.start, entity.end))
                continue
            if token.i in consumed_ent_tokens:
                continue
            lower = token.text.lower()
            if lower in self.PRONOUNS and antecedent:
                output_tokens.append(antecedent + token.whitespace_)
            else:
                output_tokens.append(token.text_with_ws)
        return "".join(output_tokens).strip()

    def _resolve_heuristic(self, document: str) -> str:
        sentences = [
            seg.strip() for seg in re.split(r"(?<=[.!?])\s+", document) if seg.strip()
        ]
        antecedent = ""
        rewritten: list[str] = []
        for sentence in sentences:
            sentence_rewritten = sentence
            if antecedent:
                pronoun_pattern = (
                    r"\b("
                    r"He|She|They|It|he|she|they|it|"
                    r"His|Her|Their|Its|his|her|their|its"
                    r")\b"
                )
                sentence_rewritten = re.sub(
                    pronoun_pattern,
                    antecedent,
                    sentence_rewritten,
                )
            candidate = self._find_latest_named_entity(sentence_rewritten)
            if candidate:
                antecedent = candidate
            rewritten.append(sentence_rewritten)
        return " ".join(rewritten)

    def _find_latest_named_entity(self, sentence: str) -> str:
        person_like = self._match_names(self._PERSON_PATTERN, sentence)
        if person_like:
            return person_like[0]
        matches = self._match_names(self._NAME_PATTERN, sentence)
        return matches[-1] if matches else ""

    @staticmethod
    def _match_names(pattern: re.Pattern[str], sentence: str) -> list[str]:
        return [
            match.group(1).strip()
            for match in pattern.finditer(sentence)
            if match.group(1).strip()
        ]

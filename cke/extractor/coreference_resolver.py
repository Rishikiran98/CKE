"""Coreference resolution for document-level extraction."""

from __future__ import annotations

import re

try:
    import spacy
except Exception:  # pragma: no cover
    spacy = None


class CoreferenceResolver:
    """Resolve pronoun references to the latest salient named entity."""

    PRONOUNS = {"he", "she", "they", "it", "him", "her", "them", "his", "its", "their"}
    _PERSON_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
    _NAME_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

    def __init__(self) -> None:
        self._nlp = self._load_spacy()

    def resolve(self, document: str) -> str:
        if not document.strip():
            return document
        resolved = self._resolve_with_spacy(document)
        if resolved is not None:
            return resolved
        return self._resolve_heuristic(document)

    def _load_spacy(self):
        if spacy is None:
            return None
        for model_name in ("en_coreference_web_trf", "en_core_web_sm"):
            model = self._try_load_spacy_model(model_name)
            if model is not None:
                return model
        return None

    @staticmethod
    def _try_load_spacy_model(model_name: str):
        try:
            return spacy.load(model_name)
        except Exception:
            # Model not present or incompatible in current runtime.
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

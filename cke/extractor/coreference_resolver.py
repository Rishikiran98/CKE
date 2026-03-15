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
            try:
                return spacy.load(model_name)
            except Exception:
                # Model not present or incompatible; try the next configured model.
                pass
        return None

    def _resolve_with_spacy(self, document: str) -> str | None:
        if self._nlp is None:
            return None
        doc = self._nlp(document)
        if not doc.ents:
            return None
        antecedent = None
        output_tokens: list[str] = []
        for token in doc:
            if token.ent_type_ and token.pos_ in {"PROPN", "NOUN"}:
                antecedent = token.text
                output_tokens.append(token.text)
                continue
            lower = token.text.lower()
            if lower in self.PRONOUNS and antecedent:
                replacement = antecedent
                if token.whitespace_:
                    replacement += token.whitespace_
                output_tokens.append(replacement)
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

    @staticmethod
    def _find_latest_named_entity(sentence: str) -> str:
        person_like = re.findall(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", sentence)
        if person_like:
            return person_like[0]
        matches = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", sentence)
        return matches[-1] if matches else ""

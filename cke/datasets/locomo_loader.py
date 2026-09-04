"""LoCoMo conversational dataset loader.

The published file is a list of ten long conversations. Each carries its own
question set in ``qa`` and its dialogue in ``conversation``, a mapping of
``session_N`` to that session's turns rather than a flat list. Every turn has
a ``dia_id`` such as ``D1:3``, and a question's ``evidence`` names the turns
that answer it by exactly those ids.

This replaces a loader written against a shape the dataset does not have. It
looked for turns under ``turns``, ``conversation``, ``messages`` or
``dialogue`` and required a list; ``conversation`` is a mapping, so against
the real file every record fell through to the degradation path and loaded
with zero documents and no question. It declared that loudly, which is the
contract working, but it could not read the dataset it names.

One item per question, not per conversation: a conversation holds around two
hundred questions and an evaluation item has one. The conversation's turns
are the documents for every question drawn from it, and the same list object
is shared between them rather than copied two hundred times.

Supporting facts name turns here, by ``dia_id``, where HotpotQA and MuSiQue
name documents by title. Each dataset labels whatever unit it retrieves, so a
recall computation has to take the identifier from the dataset rather than
assume a title.

Adversarial questions are kept and marked. Of 1986 questions, 446 are
category 5 and 444 of those carry no ``answer`` at all, only an
``adversarial_answer``: the conversation does not answer them and the
expected behaviour is to say so. That string is recorded in the item's
metadata and never as the item's answer, because scoring against it would
reward inventing the thing the question was built to catch. The remaining two
carry both, a real answer and the plausible wrong one, and are loaded with
the real answer; treating the category as the unanswerable set would have
mislabelled them.

Nine evidence entries across six conversations name no turn in their own
record: they are malformed in the published file, one pair of ids joined by a
semicolon, one bare ``D``, one with an extra colon. They are recorded as
written and declared, not repaired, because a repair is a guess at the
authors' intent and would make recall here disagree with recall computed by
anyone else on the same file.
"""

from __future__ import annotations

import json
import re
from typing import Any

from cke.datasets.base_loader import DatasetLoader
from cke.diagnostics import DegradationMixin
from cke.utils.text_cleaning import normalize_whitespace

#: ``session_4`` and its companion ``session_4_date_time``.
_SESSION_KEY = re.compile(r"^session_(\d+)$")

#: ``D4:12`` names session 4, turn 12.
_DIA_ID = re.compile(r"^D(\d+):(\d+)$")


class LoCoMoDataset(DatasetLoader, DegradationMixin):
    """Loads the published LoCoMo conversations into the CKE item shape."""

    def __init__(self, normalize_text: bool = True, strict: bool = False) -> None:
        super().__init__()
        self._init_degradation(strict)
        self.normalize_text = normalize_text

    def _clean_text(self, text: str) -> str:
        if self.normalize_text:
            return normalize_whitespace(text)
        return text.strip()

    def _conversation_documents(
        self, conversation: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Every turn of every session, in session order then turn order."""
        sessions: list[tuple[int, str]] = []
        for key in conversation:
            match = _SESSION_KEY.match(key)
            if match and isinstance(conversation[key], list):
                sessions.append((int(match.group(1)), key))
        sessions.sort()

        if not sessions:
            self._degrade(
                "a conversation record carries no session_N turn lists, so it "
                "loads with zero documents and its questions have nothing to "
                "be answered from"
            )
            return []

        documents: list[dict[str, Any]] = []
        malformed = 0
        for number, key in sessions:
            when = str(conversation.get(f"{key}_date_time", "")).strip()
            for turn in conversation[key]:
                if not isinstance(turn, dict):
                    malformed += 1
                    continue
                dia_id = str(turn.get("dia_id", "")).strip()
                text = self._clean_text(str(turn.get("text", "")))
                if not dia_id or not text:
                    malformed += 1
                    continue
                speaker = str(turn.get("speaker", "unknown")).strip()
                documents.append(
                    {
                        "doc_id": dia_id,
                        "title": f"session {number}" + (f", {when}" if when else ""),
                        "text": f"{speaker}: {text}",
                        "speaker": speaker,
                    }
                )
        if malformed:
            self._degrade(
                f"{malformed} turns carried no dia_id or no text and were "
                "dropped, so the questions drawn from this conversation are "
                "answered against fewer turns than it holds"
            )
        return documents

    @staticmethod
    def _supporting_facts(evidence: Any) -> list[list[Any]]:
        """The evidence turns, as ``[dia_id, session_number]`` pairs."""
        facts: list[list[Any]] = []
        for entry in evidence or []:
            dia_id = str(entry).strip()
            match = _DIA_ID.match(dia_id)
            facts.append([dia_id, int(match.group(1)) if match else None])
        return facts

    def conversation_items(
        self, index: int, record: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """One item per question in ``record``.

        Not ``normalize_record``: the other loaders map one record to one
        item, and one conversation here yields as many items as it has
        questions.
        """
        conversation = record.get("conversation")
        if not isinstance(conversation, dict):
            self._degrade(
                "a record carries no conversation mapping, so it loads with "
                "zero documents and its questions have nothing to be answered "
                "from"
            )
            conversation = {}

        documents = self._conversation_documents(conversation)
        known_turns = {document["doc_id"] for document in documents}
        conversation_id = str(record.get("sample_id", f"locomo_{index}"))
        speakers = [
            str(conversation.get("speaker_a", "")),
            str(conversation.get("speaker_b", "")),
        ]

        items: list[dict[str, Any]] = []
        unknown_evidence = 0
        for position, qa in enumerate(record.get("qa") or []):
            if not isinstance(qa, dict):
                continue
            question = str(qa.get("question", "")).strip()
            if not question:
                continue
            answer = qa.get("answer")
            supporting = self._supporting_facts(qa.get("evidence"))
            unknown_evidence += sum(
                1 for dia_id, _ in supporting if dia_id not in known_turns
            )
            items.append(
                {
                    "id": f"{conversation_id}::{position}",
                    "question": self._clean_text(question),
                    # str, because six of the published answers are integers.
                    # None where the dataset gives no answer at all.
                    "answer": None if answer is None else self._clean_text(str(answer)),
                    "documents": documents,
                    "supporting_facts": supporting,
                    "metadata": {
                        "conversation_id": conversation_id,
                        "category": qa.get("category"),
                        # Kept out of "answer" deliberately: see the module
                        # docstring.
                        "adversarial_answer": qa.get("adversarial_answer"),
                        "answerable": answer is not None,
                        "speakers": speakers,
                        "turns": len(documents),
                    },
                }
            )

        if unknown_evidence:
            self._degrade(
                f"{unknown_evidence} evidence ids in conversation "
                f"{conversation_id} match no turn in it, so nothing can "
                "retrieve them and recall against them is not measurable. In "
                "the published file these are malformed ids rather than "
                "missing turns, and they are recorded as written rather than "
                "guessed at"
            )
        return items

    def load(self, path: str) -> "LoCoMoDataset":
        with open(path, "r", encoding="utf-8") as f:
            raw_items = json.load(f)

        self.items = []
        for index, record in enumerate(raw_items):
            if not isinstance(record, dict):
                continue
            self.items.extend(self.conversation_items(index, record))
        return self

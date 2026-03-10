"""Shared domain models for CKE."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(slots=True)
class Entity:
    """Canonical entity stored in the graph."""

    name: str
    aliases: List[str] = field(default_factory=list)


@dataclass(slots=True)
class Statement:
    """A simple subject-relation-object statement."""

    subject: str
    relation: str
    object: str

    def as_text(self) -> str:
        return f"{self.subject} {self.relation} {self.object}"

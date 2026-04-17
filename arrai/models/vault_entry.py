from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal


@dataclass
class VaultEntry:
    """
    A technique card in Garak's vault.

    The compact representation (title + description + example + link) is
    rendered into Garak's system prompt. The full vault_path file contains
    detailed patterns, worked examples, and chaining notes.
    """

    id: str                  # URL-safe slug, e.g. "code-framing"
    title: str
    description: str         # One sentence
    example: str             # Minimal working example (plain text)
    vault_path: str          # Relative path, e.g. "techniques/seed/base64_encoding.md"
    source: Literal["seed", "garak", "human"]
    created_at: datetime
    session_id: str | None = None  # None for seed entries

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_toc_line(self) -> str:
        """Single-line entry for the system-prompt technique index."""
        return f"- {self.title} — {self.description}  →  vault://{self.vault_path}"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "example": self.example,
            "vault_path": self.vault_path,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VaultEntry":
        return cls(
            id=d["id"],
            title=d["title"],
            description=d["description"],
            example=d["example"],
            vault_path=d["vault_path"],
            source=d["source"],
            created_at=datetime.fromisoformat(d["created_at"]),
            session_id=d.get("session_id"),
        )

    @classmethod
    def create_new(
        cls,
        id: str,
        title: str,
        description: str,
        example: str,
        vault_path: str,
        source: Literal["seed", "garak", "human"] = "garak",
        session_id: str | None = None,
    ) -> "VaultEntry":
        return cls(
            id=id,
            title=title,
            description=description,
            example=example,
            vault_path=vault_path,
            source=source,
            created_at=datetime.now(timezone.utc),
            session_id=session_id,
        )

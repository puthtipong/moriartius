from __future__ import annotations

"""
VaultManager — manages Garak's technique knowledge base.

vault/
├── index.json           — list of VaultEntry objects
└── techniques/
    ├── seed/            — human-written, committed to repo
    └── discovered/      — written by Garak at runtime
"""

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arrai.models.vault_entry import VaultEntry

logger = logging.getLogger(__name__)


class VaultManager:
    """Read/write Garak's technique vault."""

    def __init__(self, vault_dir: str | Path) -> None:
        self._vault_dir = Path(vault_dir)
        self._vault_dir.mkdir(parents=True, exist_ok=True)
        (self._vault_dir / "techniques" / "seed").mkdir(parents=True, exist_ok=True)
        (self._vault_dir / "techniques" / "discovered").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_entries(self) -> list["VaultEntry"]:
        from arrai.models.vault_entry import VaultEntry

        index_path = self._vault_dir / "index.json"
        if not index_path.exists():
            return []
        try:
            data = json.loads(index_path.read_text())
            return [VaultEntry.from_dict(d) for d in data]
        except Exception as exc:
            logger.warning("Could not load vault index: %s", exc)
            return []

    def read_file(self, vault_path: str) -> str:
        """
        Read a vault file by its vault:// path.

        vault_path examples:
            vault://techniques/seed/base64_encoding.md
            techniques/seed/base64_encoding.md
        """
        clean = vault_path.replace("vault://", "").lstrip("/")
        full_path = self._vault_dir / clean
        if full_path.exists():
            return full_path.read_text(encoding="utf-8")
        return f"(File not found: {vault_path})"

    def render_toc(self) -> str:
        """
        Render all vault entries as a compact table of contents for Garak's system prompt.

        One line per entry (title, description, vault path). Seed entries first
        (alphabetical), then discovered (chronological). Garak reads full details
        on demand via read_vault_file.
        """
        entries = self.load_entries()
        if not entries:
            return ""

        seed = sorted(
            [e for e in entries if e.source == "seed"],
            key=lambda e: e.title.lower(),
        )
        discovered = sorted(
            [e for e in entries if e.source != "seed"],
            key=lambda e: e.created_at,
        )

        lines = [e.render_toc_line() for e in seed + discovered]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Write (Garak-facing)
    # ------------------------------------------------------------------

    def write_entry(
        self,
        id: str,
        title: str,
        description: str,
        example: str,
        full_content: str,
        session_id: str | None = None,
        source: str = "garak",
    ) -> "VaultEntry":
        from arrai.models.vault_entry import VaultEntry

        # Sanitise slug
        slug = re.sub(r"[^a-z0-9\-]", "-", id.lower()).strip("-") or "entry"

        # Write full vault file
        filename = f"{session_id or 'garak'}_{slug}.md" if source != "seed" else f"{slug}.md"
        subdir = "seed" if source == "seed" else "discovered"
        file_path = self._vault_dir / "techniques" / subdir / filename
        self._atomic_write(file_path, full_content)

        vault_path = f"techniques/{subdir}/{filename}"

        entry = VaultEntry.create_new(
            id=slug,
            title=title,
            description=description,
            example=example,
            vault_path=vault_path,
            source=source,
            session_id=session_id,
        )

        # Append to index.json
        entries = self.load_entries()
        # Replace existing entry with same id if present
        entries = [e for e in entries if e.id != slug]
        entries.append(entry)

        self._atomic_write(
            self._vault_dir / "index.json",
            json.dumps([e.to_dict() for e in entries], indent=2),
        )

        logger.info("Vault entry written: %s (%s)", slug, source)
        return entry

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _atomic_write(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

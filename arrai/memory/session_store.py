from __future__ import annotations

"""
SessionStore — all file I/O for a session.

Everything writes atomically (temp file → rename) to survive crashes.
All paths are relative to sessions_dir / session_id /.
"""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arrai.models.auftrag import Auftrag
    from arrai.models.mission_report import MissionReport
    from arrai.models.ooda_record import OODARecord
    from arrai.models.session_config import SessionConfig

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# target.md / plan.md templates
# ─────────────────────────────────────────────────────────────────────────────

_TARGET_MD_TEMPLATE = """\
# Target Intelligence

## Behaviors
<!-- Observed response patterns: what it does when asked X -->

## Architecture
<!-- Inferred system design: guardrails, routing, personas, etc. -->

## Dead Ends
<!-- Approaches confirmed not to work, with brief why -->

## Vulnerabilities
<!-- Confirmed exploitable behaviors or response patterns -->

## Open Questions
<!-- Unresolved hypotheses worth investigating -->
"""

_PLAN_MD_TEMPLATE = """\
# Campaign Plan

## Objective
{objective}

## Current Approach
<!-- The working hypothesis and active strategy -->

## Mission Queue
<!-- Ordered list of planned next missions (Sherlock updates this) -->

## Abandoned Approaches
<!-- What was tried and dropped, with reasoning -->
"""


# ─────────────────────────────────────────────────────────────────────────────
# SessionStore
# ─────────────────────────────────────────────────────────────────────────────

class SessionStore:
    """
    File-based persistence for one session.

    Layout:
        sessions/{session_id}/
            config.json
            target.md
            plan.md
            session.log      (JSONL — mission-level events)
            ooda.log         (JSONL — full OODARecord objects)
            missions/
                {mission_id}/
                    auftrag.json
                    trace.json
                    report.json
    """

    def __init__(self, sessions_dir: str | Path) -> None:
        self._base = Path(sessions_dir)

    # ------------------------------------------------------------------
    # Session initialisation
    # ------------------------------------------------------------------

    def init_session(self, config: "SessionConfig") -> None:
        """Create the session directory and seed all files."""
        session_dir = self._session_dir(config.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "missions").mkdir(exist_ok=True)

        # config.json
        self._atomic_write(
            session_dir / "config.json",
            json.dumps(config.to_dict(), indent=2),
        )

        # target.md — seed with whitebox info if provided
        target_md = _TARGET_MD_TEMPLATE
        if config.whitebox_seed:
            target_md = target_md.replace(
                "<!-- Inferred system design: guardrails, routing, personas, etc. -->",
                config.whitebox_seed,
            )
        if config.further_context:
            target_md += f"\n## Initial Context\n{config.further_context}\n"
        self._atomic_write(session_dir / "target.md", target_md)

        # plan.md
        plan_md = _PLAN_MD_TEMPLATE.format(objective=config.objective)
        self._atomic_write(session_dir / "plan.md", plan_md)

        # Empty logs
        self._atomic_write(session_dir / "session.log", "")
        self._atomic_write(session_dir / "ooda.log", "")

        logger.info("Session %s initialised at %s", config.session_id, session_dir)

    def load_config(self, session_id: str) -> dict:
        path = self._session_dir(session_id) / "config.json"
        return json.loads(path.read_text())

    # ------------------------------------------------------------------
    # target.md / plan.md
    # ------------------------------------------------------------------

    def read_target_md(self, session_id: str) -> str:
        return self._read(self._session_dir(session_id) / "target.md")

    def write_target_md(self, session_id: str, content: str) -> None:
        self._atomic_write(self._session_dir(session_id) / "target.md", content)

    def read_plan_md(self, session_id: str) -> str:
        return self._read(self._session_dir(session_id) / "plan.md")

    def write_plan_md(self, session_id: str, content: str) -> None:
        self._atomic_write(self._session_dir(session_id) / "plan.md", content)

    def write_session_report(self, session_id: str, content: str) -> None:
        self._atomic_write(self._session_dir(session_id) / "session_report.md", content)

    def read_session_report(self, session_id: str) -> str:
        return self._read(self._session_dir(session_id) / "session_report.md")

    # ------------------------------------------------------------------
    # Mission persistence
    # ------------------------------------------------------------------

    def save_mission(
        self,
        session_id: str,
        auftrag: "Auftrag",
        report: "MissionReport",
    ) -> None:
        mission_dir = self._mission_dir(session_id, auftrag.mission_id)
        mission_dir.mkdir(parents=True, exist_ok=True)

        self._atomic_write(
            mission_dir / "auftrag.json",
            json.dumps(auftrag.to_dict(), indent=2),
        )
        self._atomic_write(
            mission_dir / "trace.json",
            json.dumps(report.conversation_trace.to_dict(), indent=2),
        )
        self._atomic_write(
            mission_dir / "report.json",
            json.dumps(report.to_dict(), indent=2),
        )

    def load_all_reports(self, session_id: str) -> list["MissionReport"]:
        from arrai.models.mission_report import MissionReport

        reports = []
        missions_dir = self._session_dir(session_id) / "missions"
        if not missions_dir.exists():
            return reports

        for mission_dir in sorted(missions_dir.iterdir()):
            report_file = mission_dir / "report.json"
            if report_file.exists():
                try:
                    reports.append(MissionReport.from_dict(json.loads(report_file.read_text())))
                except Exception as exc:
                    logger.warning("Could not load report %s: %s", report_file, exc)
        return reports

    # ------------------------------------------------------------------
    # OODA log
    # ------------------------------------------------------------------

    def append_ooda_record(self, session_id: str, record: "OODARecord") -> None:
        line = json.dumps(record.to_dict()) + "\n"
        self._append(self._session_dir(session_id) / "ooda.log", line)

    def get_reasoning_chain(self, session_id: str) -> str:
        """
        Return the Orient+Decide chain from all past OODA cycles.
        Fed back to Sherlock for continuity of reasoning.
        """
        from arrai.models.ooda_record import OODARecord

        path = self._session_dir(session_id) / "ooda.log"
        if not path.exists():
            return ""

        entries = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = OODARecord.from_dict(json.loads(line))
                entries.append(record.reasoning_chain_entry())
            except Exception:
                pass

        if len(entries) > self._REASONING_CHAIN_WINDOW:
            dropped = len(entries) - self._REASONING_CHAIN_WINDOW
            entries = entries[-self._REASONING_CHAIN_WINDOW:]
            entries.insert(
                0,
                f"[{dropped} earlier reasoning entries omitted — "
                "digested into target.md / plan.md]",
            )

        return "\n".join(entries)

    # ------------------------------------------------------------------
    # Session log
    # ------------------------------------------------------------------

    def append_mission_log_entry(
        self,
        session_id: str,
        report: "MissionReport",
    ) -> None:
        entry = {
            "type": "mission_complete",
            "mission_id": report.mission_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "terminal_condition": report.terminal_condition,
            "garak_score": report.garak_score,
            "scorer_score": report.scorer_score,
            "techniques_used": report.techniques_used,
            "garak_insights": report.garak_insights,
            "turn_count": len([
                m for m in report.conversation_trace.messages
                if m.role == "user"
            ]),
        }
        self._append(
            self._session_dir(session_id) / "session.log",
            json.dumps(entry) + "\n",
        )

    def append_ooda_log_entry(self, session_id: str, record: "OODARecord") -> None:
        entry = {
            "type": "ooda_complete",
            "cycle_id": record.cycle_id,
            "timestamp": record.timestamp.isoformat(),
            "decide_summary": record.decide[:200],
            "action_type": record.action_type,
        }
        self._append(
            self._session_dir(session_id) / "session.log",
            json.dumps(entry) + "\n",
        )

    def append_observation(
        self, session_id: str, mission_id: str, observation: str
    ) -> None:
        entry = {
            "type": "observation",
            "mission_id": mission_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "observation": observation,
        }
        self._append(
            self._session_dir(session_id) / "session.log",
            json.dumps(entry) + "\n",
        )

    # ------------------------------------------------------------------
    # Context helpers (used by agents)
    # ------------------------------------------------------------------

    def get_mission_summaries(self, session_id: str) -> list[str]:
        """One-line summary per past mission — fed to Garak as adaptive context."""
        reports = self.load_all_reports(session_id)
        return [r.summary_line() for r in reports]

    def get_observations_for_mission(self, session_id: str, mission_id: str) -> list[str]:
        """
        Return all log_observation entries Garak wrote during a specific mission.
        Read from session.log, filtered by mission_id.
        """
        path = self._session_dir(session_id) / "session.log"
        if not path.exists():
            return []
        observations = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if (
                    entry.get("type") == "observation"
                    and entry.get("mission_id") == mission_id
                ):
                    observations.append(entry["observation"])
            except Exception:
                pass
        return observations

    # Windowing constants: keep the last N missions as full traces;
    # anything older is collapsed to a one-line summary.
    _TRACE_FULL_WINDOW = 10
    _TRACE_SUMMARY_THRESHOLD = 20   # only activate windowing above this count

    # Reasoning chain windowing: keep last N OODA cycles' Orient+Decide.
    # Older reasoning is already digested into target.md / plan.md.
    _REASONING_CHAIN_WINDOW = 20

    def get_all_traces(self, session_id: str) -> str:
        """
        Full conversation traces for all missions — fed to Sherlock.

        Windowing: when a session has more than _TRACE_SUMMARY_THRESHOLD missions,
        only the most recent _TRACE_FULL_WINDOW are returned as full traces.
        Older missions are collapsed to a single summary line each so that
        Sherlock's context stays manageable on long sessions.
        """
        reports = self.load_all_reports(session_id)
        if not reports:
            return ""

        sections: list[str] = []

        if len(reports) > self._TRACE_SUMMARY_THRESHOLD:
            cutoff = len(reports) - self._TRACE_FULL_WINDOW
            old_reports = reports[:cutoff]
            recent_reports = reports[cutoff:]

            # Compact block for older missions
            summary_lines = [
                f"### Older Missions — summaries only ({len(old_reports)} missions)",
                "",
            ]
            for r in old_reports:
                summary_lines.append(f"- {r.summary_line()}")
            sections.append("\n".join(summary_lines))
        else:
            recent_reports = reports

        # Full traces for recent missions
        for report in recent_reports:
            lines = [
                f"### Mission {report.mission_id[:8]} — {report.terminal_condition}",
                f"Score: garak={report.garak_score:.2f} scorer={report.scorer_score:.2f}",
            ]
            if report.scorer_rationale:
                lines.append(f"Scorer: {report.scorer_rationale}")
            lines.append("")
            for msg in report.conversation_trace.messages:
                prefix = "TESTER" if msg.role == "user" else "TARGET"
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                lines.append(f"[{prefix}]: {content}")
            observations = self.get_observations_for_mission(session_id, report.mission_id)
            if observations:
                lines.append("")
                lines.append("Garak field observations:")
                for o in observations:
                    lines.append(f"  • {o}")
            sections.append("\n".join(lines))

        return "\n\n---\n\n".join(sections)

    # ------------------------------------------------------------------
    # Session listing
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[dict]:
        sessions = []
        if not self._base.exists():
            return sessions
        for session_dir in self._base.iterdir():
            config_file = session_dir / "config.json"
            if config_file.exists():
                try:
                    cfg = json.loads(config_file.read_text())
                    reports = self.load_all_reports(cfg["session_id"])
                    sessions.append({
                        "session_id": cfg["session_id"],
                        "objective": cfg.get("objective", ""),
                        "mission_count": len(reports),
                        "mode": cfg.get("mode", "autonomous"),
                    })
                except Exception:
                    pass
        return sessions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session_dir(self, session_id: str) -> Path:
        return self._base / session_id

    def _mission_dir(self, session_id: str, mission_id: str) -> Path:
        return self._session_dir(session_id) / "missions" / mission_id

    def _atomic_write(self, path: Path, content: str) -> None:
        """Write content to path atomically using a temp file + rename."""
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

    def _append(self, path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)

    def _read(self, path: Path) -> str:
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

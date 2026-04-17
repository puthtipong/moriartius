from __future__ import annotations

"""
SessionRunner — the main async loop that orchestrates Sherlock ↔ Garak.

Each iteration:
  1. Sherlock runs an OODA cycle (reads full context, decides action)
  2. Apply any target.md / plan.md updates
  3. Dispatch Garak with the Auftrag
  4. Scorer evaluates the conversation
  5. Persist everything
  6. Repeat until complete / max_missions
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Awaitable

if TYPE_CHECKING:
    pass  # Session referenced in on_session_ready signature


def _to_str(value: Any) -> str:
    """
    Coerce any LLM-returned value to a plain string.

    Sherlock occasionally returns dict/list for fields that should be strings
    (e.g. session_summary, observe, garak_insights).  This ensures we always
    produce a str before passing values to str.join() or writing to disk.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False)

from arrai.agents.garak import GarakAgent
from arrai.agents.sherlock import SherlockAgent
from arrai.agents.scorer import ScorerModel
from arrai.memory.session_store import SessionStore
from arrai.memory.vault import VaultManager
from arrai.models.mission_report import MissionReport
from arrai.models.ooda_record import OODARecord
from arrai.models.session_config import SessionConfig
from arrai.targets import build_target, Target
from arrai.tools.registry import build_registry
from arrai.vault_seeder import seed_vault_if_empty

logger = logging.getLogger(__name__)

# Type alias: async callable that receives a dict and broadcasts to GUI/log
Emitter = Callable[[dict], Awaitable[None]]


async def _noop_emit(event: dict) -> None:
    """Default emitter: log to debug."""
    logger.debug("SSE event: %s", event.get("type"))


# ─────────────────────────────────────────────────────────────────────────────
# Session  — the shared context object threaded through all components
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Session:
    """
    Container for all live objects in a running session.

    Passed to Sherlock, Garak, and tool handlers so they can
    read/write session files and emit SSE events.
    """

    config: SessionConfig
    target: Target
    store: SessionStore
    vault: VaultManager
    emit: Emitter = field(default=_noop_emit)

    # HITL: when mode==hitl, this future is set when Sherlock pauses
    _hitl_future: asyncio.Future | None = field(default=None, init=False, repr=False)

    async def wait_for_hitl(self) -> dict:
        """Block until a HITL response arrives (approve/revise/reject)."""
        loop = asyncio.get_event_loop()
        self._hitl_future = loop.create_future()
        return await self._hitl_future

    def resolve_hitl(self, response: dict) -> None:
        """Called by the API handler when the human responds."""
        if self._hitl_future and not self._hitl_future.done():
            self._hitl_future.set_result(response)


# ─────────────────────────────────────────────────────────────────────────────
# SessionRunner
# ─────────────────────────────────────────────────────────────────────────────

class SessionRunner:
    """
    Orchestrates the full Sherlock ↔ Garak session loop.

    Usage:
        runner = SessionRunner(config, sessions_dir, vault_dir)
        await runner.run()      # new session
        await runner.resume()   # resume from disk
    """

    def __init__(
        self,
        config: SessionConfig,
        sessions_dir: str | Path = "./sessions",
        vault_dir: str | Path = "./vault",
        emit: Emitter | None = None,
        on_session_ready: Callable[["Session"], None] | None = None,
    ) -> None:
        self._config = config
        self._store = SessionStore(sessions_dir)
        self._vault = VaultManager(vault_dir)
        self._emit = emit or _noop_emit
        self._on_session_ready = on_session_ready

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start a brand-new session."""
        # Build target first — validates config/credentials before creating any files
        target = build_target(self._config.target_config)
        self._store.init_session(self._config)
        # Seed vault on first run
        seeded = seed_vault_if_empty(self._vault)
        if seeded:
            await self._emit({"type": "vault_seeded", "entries": seeded})
        await self._emit({"type": "session_init", "session_id": self._config.session_id})
        await self._main_loop(last_report=None, target=target)

    async def resume(self) -> None:
        """Resume an existing session from disk."""
        target = build_target(self._config.target_config)
        # Seed vault if it was never populated (e.g. fresh vault_dir)
        seed_vault_if_empty(self._vault)
        reports = self._store.load_all_reports(self._config.session_id)
        last_report = reports[-1] if reports else None
        logger.info(
            "Resuming session %s from mission %d",
            self._config.session_id,
            len(reports),
        )
        await self._emit({"type": "session_resume", "session_id": self._config.session_id})
        await self._main_loop(last_report=last_report, target=target)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def _main_loop(self, last_report: MissionReport | None, target: Target) -> None:
        config = self._config
        tool_registry = build_registry()
        sherlock = SherlockAgent(config)
        garak = GarakAgent(config, tool_registry)
        scorer = ScorerModel(config)

        session = Session(
            config=config,
            target=target,
            store=self._store,
            vault=self._vault,
            emit=self._emit,
        )
        if self._on_session_ready:
            self._on_session_ready(session)

        mission_count = len(self._store.load_all_reports(config.session_id))

        while mission_count < config.max_missions:

            # ── 1. Sherlock OODA ──────────────────────────────────────
            await self._emit({"type": "ooda_start", "session_id": config.session_id})

            ooda: OODARecord = await sherlock.run_ooda_cycle(last_report, session)

            await self._emit({
                "type": "ooda_complete",
                "cycle_id": ooda.cycle_id,
                "action_type": ooda.action_type,
                "observe": ooda.observe,
                "orient": ooda.orient,
                "decide": ooda.decide,
            })

            # Persist OODA record
            self._store.append_ooda_record(config.session_id, ooda)
            self._store.append_ooda_log_entry(config.session_id, ooda)

            # ── 2. Apply memory updates ───────────────────────────────
            # "NO_CHANGE: ..." is the sentinel Sherlock uses when nothing is new.
            # Anything else is treated as the full updated file content.
            target_md_val = ooda.update_target_md
            if target_md_val and not isinstance(target_md_val, str):
                target_md_val = json.dumps(target_md_val, indent=2)
            if target_md_val and not target_md_val.strip().startswith("NO_CHANGE"):
                self._store.write_target_md(config.session_id, target_md_val)
                await self._emit({"type": "target_md_updated", "session_id": config.session_id})

            plan_md_val = ooda.update_plan_md
            if plan_md_val and not isinstance(plan_md_val, str):
                plan_md_val = json.dumps(plan_md_val, indent=2)
            if plan_md_val and not plan_md_val.strip().startswith("NO_CHANGE"):
                self._store.write_plan_md(config.session_id, plan_md_val)
                await self._emit({"type": "plan_md_updated", "session_id": config.session_id})

            # ── 3. Handle action ──────────────────────────────────────
            if ooda.action_type == "complete":
                logger.info("Sherlock declared session complete.")
                self._write_session_report(ooda, session)
                await self._emit({
                    "type": "session_complete",
                    "session_id": config.session_id,
                    "objective_achieved": True,
                    "session_summary": _to_str(ooda.session_summary),
                })
                return

            if ooda.action_type == "pause_for_hitl" and config.mode == "hitl":
                ooda = await self._handle_hitl(ooda, session)

            if ooda.action_type == "session_reset":
                await target.reset_async()
                await self._emit({"type": "session_reset", "session_id": config.session_id})

            if ooda.auftrag is None:
                logger.warning("Sherlock action %r has no Auftrag — skipping.", ooda.action_type)
                continue

            auftrag = ooda.auftrag

            # ── 4. Dispatch Garak ─────────────────────────────────────
            await self._emit({
                "type": "mission_start",
                "mission_id": auftrag.mission_id,
                "objective": auftrag.objective,
                "turn_budget": auftrag.turn_budget,
            })

            report: MissionReport = await garak.run_mission(auftrag, session)

            # ── 5. Score ──────────────────────────────────────────────
            score, rationale = await scorer.score_async(
                report.conversation_trace,
                auftrag.success_criteria,
            )
            report.scorer_score = score
            report.scorer_rationale = rationale

            await self._emit({
                "type": "scorer_result",
                "mission_id": auftrag.mission_id,
                "score": score,
                "rationale": rationale,
            })

            # ── 6. Persist ────────────────────────────────────────────
            self._store.save_mission(config.session_id, auftrag, report)
            self._store.append_mission_log_entry(config.session_id, report)

            await self._emit({
                "type": "mission_complete",
                "mission_id": report.mission_id,
                "terminal_condition": report.terminal_condition,
                "garak_score": report.garak_score,
                "scorer_score": report.scorer_score,
            })

            # ── 7. Feed report back to Sherlock on next cycle ─────────
            # No automatic termination here. Sherlock is the only one who
            # can declare the OVERALL objective complete. Mission scorer
            # scores are signal for Sherlock, not termination triggers.
            last_report = report
            mission_count += 1

        # Max missions reached — give Sherlock one final cycle to summarise
        logger.info("Max missions (%d) reached. Running final Sherlock review.", config.max_missions)
        final_ooda = await sherlock.run_ooda_cycle(last_report, session)
        self._store.append_ooda_record(config.session_id, final_ooda)
        self._write_session_report(final_ooda, session)
        await self._emit({
            "type": "session_complete",
            "session_id": config.session_id,
            "objective_achieved": False,
            "reason": "max_missions_reached",
            "session_summary": final_ooda.session_summary or "",
        })

    # ------------------------------------------------------------------
    # Session report
    # ------------------------------------------------------------------

    def _write_session_report(self, ooda: "OODARecord", session: "Session") -> None:
        """
        Write session_report.md to the session directory.

        Contains Sherlock's final analysis: what worked, what failed,
        the winning technique chain, target weaknesses, and follow-up ideas.
        """
        config = self._config
        reports = self._store.load_all_reports(config.session_id)

        lines = [
            f"# ArrAI Session Report",
            f"",
            f"**Session ID:** {config.session_id}",
            f"**Objective:** {config.objective}",
            f"**Target:** {config.target_config.target_type} / "
            f"{config.target_config.params.get('model', '')}",
            f"**Missions run:** {len(reports)}",
            f"**Outcome:** {'✅ Objective achieved' if ooda.action_type == 'complete' else '⏹ Max missions reached'}",
            f"",
            f"---",
            f"",
            f"## Sherlock's Final Analysis",
            f"",
            _to_str(ooda.session_summary) or "(No summary provided.)",
            f"",
            f"---",
            f"",
            f"## Mission Log",
            f"",
        ]

        for r in reports:
            icon = "✅" if r.terminal_condition == "success" else (
                "💡" if r.terminal_condition == "discovery" else "❌"
            )
            lines.append(
                f"- {icon} `{r.mission_id[:8]}` — **{r.terminal_condition}** | "
                f"garak={r.garak_score:.2f} scorer={r.scorer_score:.2f} | "
                f"techniques: {r.techniques_used}"
            )
            if r.garak_insights:
                lines.append(f"  > {r.garak_insights[:150]}")

        lines += ["", "---", "", "## target.md at close", ""]
        lines.append(self._store.read_target_md(config.session_id))

        self._store.write_session_report(config.session_id, "\n".join(lines))
        logger.info("Session report written for %s.", config.session_id)

    # ------------------------------------------------------------------
    # HITL handling
    # ------------------------------------------------------------------

    async def _handle_hitl(self, ooda: OODARecord, session: Session) -> OODARecord:
        """
        Pause and wait for human response.

        Returns the (possibly revised) OODARecord to proceed with.
        """
        await self._emit({
            "type": "hitl_pause",
            "session_id": self._config.session_id,
            "cycle_id": ooda.cycle_id,
            "auftrag": ooda.auftrag.to_dict() if ooda.auftrag else None,
            "decide": ooda.decide,
        })

        response = await session.wait_for_hitl()
        action = response.get("action", "approve")

        # Notify the frontend that the pause is over
        await self._emit({
            "type": "hitl_resume",
            "session_id": self._config.session_id,
            "action": action,
        })

        if action == "approve":
            return ooda

        if action == "revise" and ooda.auftrag:
            revision = response.get("revision", {})
            auftrag = ooda.auftrag
            # Apply any field overrides from the human
            for field_name, value in revision.items():
                if hasattr(auftrag, field_name):
                    setattr(auftrag, field_name, value)
            return ooda

        if action == "reject":
            # Human rejected — clear the auftrag so the loop skips the dispatch
            # and runs Sherlock again on the next iteration with no new mission.
            logger.info("HITL rejection — Sherlock will re-plan on next cycle.")
            ooda.auftrag = None
            return ooda

        return ooda

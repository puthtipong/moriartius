from __future__ import annotations

"""
ArrAI FastAPI application.

Serves the single-page GUI and provides a REST + SSE API for:
  - Listing / starting sessions
  - Streaming live session events (SSE)
  - Reading session intel (target.md, plan.md, report)
  - Browsing the technique vault
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


# ─────────────────────────────────────────────────────────────────────────────
# Request / response models  (must be at module scope so FastAPI can resolve
# the annotation strings produced by "from __future__ import annotations")
# ─────────────────────────────────────────────────────────────────────────────

class StartSessionRequest(BaseModel):
    objective: str
    further_context: str = ""
    target_type: str = "openai"
    # LLM target fields (openai / anthropic / azure_openai)
    target_model: str = "gpt-4o-mini"
    target_system_prompt: str = ""
    # Arbitrary extra params for any target type (JSON string).
    # Merged over target_model / target_system_prompt when provided.
    target_params_json: str = ""
    sherlock_model: str = "gpt-4o"
    sherlock_effort: str = "medium"
    garak_model: str = "gpt-4o"
    garak_effort: str = "none"
    scorer_model: str = "gpt-4o-mini"
    max_missions: int = 10
    turn_budget: int = 8
    mode: str = "autonomous"


class SetKeysRequest(BaseModel):
    # Each field is optional — only non-empty strings are applied.
    # Keys are written to os.environ for this server process only.
    # They are NEVER persisted to disk.
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    extra_key_name: str = ""   # arbitrary env-var name
    extra_key_value: str = ""


class HitlRequest(BaseModel):
    action: str          # "approve" | "revise" | "reject"
    revision: dict = {}  # optional edits to the Auftrag when action=="revise"


# ─────────────────────────────────────────────────────────────────────────────
# Per-session state (in-memory, current server run only)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ManagedSession:
    """Tracks a running (or just-completed) session started in this process."""
    session_id: str
    event_log: list[dict] = field(default_factory=list)
    status: str = "running"   # running | complete | error
    task: Any = field(default=None, repr=False)
    session_obj: Any = field(default=None, repr=False)  # live Session; set by on_session_ready


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app(
    sessions_dir: Path = Path("./sessions"),
    vault_dir: Path = Path("./vault"),
) -> FastAPI:
    """
    Create and return the FastAPI application.

    Args:
        sessions_dir: Where session files are stored.
        vault_dir:    Where technique vault files are stored.
    """
    sessions_dir = sessions_dir.resolve()
    vault_dir = vault_dir.resolve()

    # Per-process session registry
    managed: dict[str, ManagedSession] = {}

    # ── Lifespan ──────────────────────────────────────────────────────────
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Seed vault on startup (idempotent)
        from arrai.memory.vault import VaultManager
        from arrai.vault_seeder import seed_vault_if_empty
        try:
            seed_vault_if_empty(VaultManager(vault_dir))
        except Exception as exc:
            logger.warning("Vault seeding failed at startup: %s", exc)
        yield

    app = FastAPI(title="ArrAI", docs_url=None, redoc_url=None, lifespan=lifespan)

    # ── Static files ──────────────────────────────────────────────────────
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    # ─────────────────────────────────────────────────────────────────────
    # Pages
    # ─────────────────────────────────────────────────────────────────────

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(_STATIC_DIR / "index.html")

    # ─────────────────────────────────────────────────────────────────────
    # Sessions API
    # ─────────────────────────────────────────────────────────────────────

    @app.get("/api/sessions")
    async def list_sessions():
        from arrai.memory.session_store import SessionStore
        store = SessionStore(sessions_dir)
        sessions = store.list_sessions()

        for s in sessions:
            sid = s["session_id"]
            # Status: prefer in-memory, then check report file
            if sid in managed:
                s["status"] = managed[sid].status
            elif (sessions_dir / sid / "session_report.md").exists():
                s["status"] = "complete"
            else:
                s["status"] = "incomplete"

            # Target info for display
            try:
                cfg = store.load_config(sid)
                tc = cfg.get("target_config", {})
                ttype = tc.get("target_type", "")
                tparams = tc.get("params", {})
                # Prefer model name for LLM targets; fall back to type label
                model = tparams.get("model", "")
                s["target_type"] = ttype
                s["target_model"] = model or ttype
                s["sherlock_model"] = cfg.get("sherlock_model", "")
            except Exception:
                s["target_type"] = ""
                s["target_model"] = ""
                s["sherlock_model"] = ""

        # Sort: running first, then by session_id (proxy for recency)
        sessions.sort(
            key=lambda s: (0 if s["status"] == "running" else 1, s["session_id"]),
            reverse=False,
        )
        return sessions

    @app.post("/api/sessions")
    async def start_session(req: StartSessionRequest):
        import json as _json
        from arrai.models.session_config import SessionConfig, TargetConfig

        # Build base target params
        llm_types = {"openai", "anthropic", "azure_openai"}
        if req.target_type in llm_types:
            target_params: dict = {"model": req.target_model}
            if req.target_system_prompt:
                target_params["system_prompt"] = req.target_system_prompt
        else:
            target_params = {}

        # Merge in any extra params from the JSON field
        if req.target_params_json and req.target_params_json.strip():
            try:
                extra = _json.loads(req.target_params_json)
                if isinstance(extra, dict):
                    target_params.update(extra)
            except _json.JSONDecodeError as exc:
                from fastapi import HTTPException
                raise HTTPException(status_code=422, detail=f"target_params_json is not valid JSON: {exc}")

        config = SessionConfig.create(
            target_config=TargetConfig(
                target_type=req.target_type,
                params=target_params,
            ),
            objective=req.objective,
            further_context=req.further_context or None,
            mode=req.mode,
            max_missions=req.max_missions,
            default_turn_budget=req.turn_budget,
            sherlock_model=req.sherlock_model,
            sherlock_effort=req.sherlock_effort,
            garak_model=req.garak_model,
            garak_effort=req.garak_effort,
            scorer_model=req.scorer_model,
        )

        sid = config.session_id
        ms = ManagedSession(session_id=sid)
        managed[sid] = ms
        _launch_runner(ms, config)
        return {"session_id": sid}

    def _launch_runner(ms: ManagedSession, config, *, resume: bool = False) -> None:
        """
        Shared helper: wire up emit + on_session_ready, create task.
        Called by both start_session and resume_session.
        """
        sid = ms.session_id

        async def emit(event: dict) -> None:
            ms.event_log.append(event)

        def on_session_ready(session_obj) -> None:
            ms.session_obj = session_obj

        async def run_it() -> None:
            try:
                from arrai.runner import SessionRunner
                runner = SessionRunner(
                    config=config,
                    sessions_dir=sessions_dir,
                    vault_dir=vault_dir,
                    emit=emit,
                    on_session_ready=on_session_ready,
                )
                if resume:
                    await runner.resume()
                else:
                    await runner.run()
                ms.status = "complete"
            except Exception as exc:
                logger.exception("Session %s failed: %s", sid, exc)
                ms.event_log.append({"type": "error", "message": str(exc)})
                ms.status = "error"
            finally:
                ms.event_log.append({"type": "stream_done"})

        ms.task = asyncio.create_task(run_it())

    @app.post("/api/sessions/{session_id}/resume")
    async def resume_session(session_id: str):
        """Resume an incomplete session from disk."""
        from arrai.memory.session_store import SessionStore
        from arrai.models.session_config import SessionConfig

        session_dir = sessions_dir / session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")

        # Already running in this process?
        existing = managed.get(session_id)
        if existing and existing.status == "running":
            return {"session_id": session_id}  # idempotent — just navigate to it

        store = SessionStore(sessions_dir)
        try:
            cfg_dict = store.load_config(session_id)
            config = SessionConfig.from_dict(cfg_dict)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not load session config: {exc}")

        ms = ManagedSession(session_id=session_id)
        managed[session_id] = ms
        _launch_runner(ms, config, resume=True)
        return {"session_id": session_id}

    @app.post("/api/sessions/{session_id}/abort")
    async def abort_session(session_id: str):
        """
        Request a graceful abort: the session stops after the current mission
        completes rather than being killed mid-execution.
        """
        ms = managed.get(session_id)
        if not ms:
            raise HTTPException(status_code=404, detail="Session not found in this server run")
        if ms.status != "running":
            raise HTTPException(status_code=409, detail=f"Session is not running (status={ms.status})")
        if not ms.session_obj:
            raise HTTPException(status_code=409, detail="Session object not ready yet")
        ms.session_obj.request_abort()
        return {"ok": True, "message": "Abort requested — session will stop after current mission."}

    @app.post("/api/sessions/{session_id}/hitl")
    async def hitl_respond(session_id: str, req: HitlRequest):
        """
        Deliver a Human-in-the-Loop decision for a paused session.

        action: "approve" | "revise" | "reject"
        revision: dict of Auftrag field overrides (only used for "revise")
        """
        ms = managed.get(session_id)
        if not ms:
            raise HTTPException(status_code=404, detail="Session not found in this server run")
        if ms.status != "running":
            raise HTTPException(status_code=409, detail=f"Session is not running (status={ms.status})")
        if not ms.session_obj:
            raise HTTPException(status_code=409, detail="Session object not ready yet")
        ms.session_obj.resolve_hitl({"action": req.action, "revision": req.revision})
        return {"ok": True}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        from arrai.memory.session_store import SessionStore
        store = SessionStore(sessions_dir)

        session_dir = sessions_dir / session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")

        try:
            cfg = store.load_config(session_id)
        except Exception:
            raise HTTPException(status_code=404, detail="Session config not found")

        ms = managed.get(session_id)
        if ms:
            status = ms.status
        elif (session_dir / "session_report.md").exists():
            status = "complete"
        else:
            status = "incomplete"

        reports = store.load_all_reports(session_id)
        has_stream = session_id in managed

        return {
            "session_id": session_id,
            "config": cfg,
            "status": status,
            "mission_count": len(reports),
            "has_stream": has_stream,
            "target_md": store.read_target_md(session_id),
            "plan_md": store.read_plan_md(session_id),
            "report": store.read_session_report(session_id),
        }

    @app.get("/api/sessions/{session_id}/intel")
    async def get_intel(session_id: str):
        """Lightweight endpoint to refresh intel docs without reloading everything."""
        from arrai.memory.session_store import SessionStore
        store = SessionStore(sessions_dir)
        return {
            "target_md": store.read_target_md(session_id),
            "plan_md": store.read_plan_md(session_id),
            "report": store.read_session_report(session_id),
        }

    @app.get("/api/sessions/{session_id}/stream")
    async def stream_session(session_id: str):
        """
        SSE endpoint.

        - If session is in memory: replay full event log, then tail live events.
        - If session is not in memory (previous run): send a single 'not_in_memory' event.

        Polls the event_log list at 100ms intervals — simple and race-condition-free.
        """
        ms = managed.get(session_id)

        async def generate():
            if ms is None:
                yield f"data: {json.dumps({'type': 'not_in_memory'})}\n\n"
                yield f"data: {json.dumps({'type': 'stream_done'})}\n\n"
                return

            index = 0
            while True:
                # Drain any new events
                while index < len(ms.event_log):
                    yield f"data: {json.dumps(ms.event_log[index])}\n\n"
                    index += 1

                # If session finished and we've drained everything, close
                if ms.status in ("complete", "error"):
                    # One final drain pass
                    while index < len(ms.event_log):
                        yield f"data: {json.dumps(ms.event_log[index])}\n\n"
                        index += 1
                    break

                await asyncio.sleep(0.1)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.get("/api/sessions/{session_id}/export")
    async def export_session(session_id: str):
        """Download the full session directory as a zip archive."""
        import io
        import zipfile
        from fastapi.responses import Response

        session_dir = sessions_dir / session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(session_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(session_dir))
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=arrai-{session_id[:8]}.zip"},
        )

    @app.get("/api/sessions/{session_id}/replay")
    async def replay_session(session_id: str):
        """
        Reconstruct the full event sequence for a historical session from disk.

        Reads ooda.log (JSONL) + missions/*/report.json + missions/*/trace.json
        and returns them as an ordered list of the same event dicts that the
        live SSE stream would have emitted.  The frontend processes these
        identically to live events, giving full timeline replay for any session.
        """
        session_dir = sessions_dir / session_id
        if not session_dir.exists():
            raise HTTPException(status_code=404, detail="Session not found")

        events: list[dict] = []

        # ── Load OODA records ────────────────────────────────────────────
        ooda_log = session_dir / "ooda.log"
        ooda_records: list[dict] = []
        if ooda_log.exists():
            for raw in ooda_log.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if raw:
                    try:
                        ooda_records.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass

        # ── Load mission reports + traces keyed by mission_id ─────────────
        missions_dir = session_dir / "missions"
        reports: dict[str, dict] = {}
        traces: dict[str, list[dict]] = {}
        if missions_dir.exists():
            for mission_dir in sorted(missions_dir.iterdir()):
                r_file = mission_dir / "report.json"
                t_file = mission_dir / "trace.json"
                if r_file.exists():
                    try:
                        r = json.loads(r_file.read_text(encoding="utf-8"))
                        reports[r["mission_id"]] = r
                    except Exception:
                        pass
                if t_file.exists():
                    try:
                        t = json.loads(t_file.read_text(encoding="utf-8"))
                        traces[t["mission_id"]] = t.get("messages", [])
                    except Exception:
                        pass

        # ── Reconstruct events ────────────────────────────────────────────
        def _s(v: object) -> str:
            if isinstance(v, str):
                return v
            if v is None:
                return ""
            return json.dumps(v, ensure_ascii=False)

        for ooda in ooda_records:
            cycle_id = ooda.get("cycle_id", "")
            action_type = ooda.get("action_type", "")

            events.append({
                "type": "ooda_start",
                "cycle_id": cycle_id,
            })
            events.append({
                "type": "ooda_complete",
                "cycle_id": cycle_id,
                "action_type": action_type,
                "observe": _s(ooda.get("observe")),
                "orient": _s(ooda.get("orient")),
                "decide": _s(ooda.get("decide")),
            })

            if action_type == "mission" and ooda.get("auftrag"):
                auftrag = ooda["auftrag"]
                mission_id = auftrag.get("mission_id", "")

                events.append({
                    "type": "mission_start",
                    "cycle_id": cycle_id,
                    "mission_id": mission_id,
                    "objective": _s(auftrag.get("objective")),
                    "success_criteria": _s(auftrag.get("success_criteria")),
                    "turn_budget": auftrag.get("turn_budget", 8),
                })

                # Conversation turns from trace
                turn = 0
                for msg in traces.get(mission_id, []):
                    if msg.get("role") == "user":
                        turn += 1
                        events.append({
                            "type": "turn",
                            "mission_id": mission_id,
                            "turn": turn,
                            "prompt": _s(msg.get("content")),
                        })
                    elif msg.get("role") == "assistant":
                        events.append({
                            "type": "turn_response",
                            "mission_id": mission_id,
                            "turn": turn,
                            "response": _s(msg.get("content")),
                        })

                # Mission outcome
                rep = reports.get(mission_id, {})
                events.append({
                    "type": "mission_complete",
                    "cycle_id": cycle_id,
                    "mission_id": mission_id,
                    "terminal_condition": rep.get("terminal_condition", "unknown"),
                    "garak_score": rep.get("garak_score", 0.0),
                    "turn_count": turn,
                    "total_budget": auftrag.get("turn_budget", 8),
                })
                if rep.get("scorer_score") is not None:
                    events.append({
                        "type": "scorer_result",
                        "mission_id": mission_id,
                        "score": rep.get("scorer_score", 0.0),
                        "rationale": _s(rep.get("scorer_rationale")),
                    })

        # Session-complete event (if it finished)
        if (session_dir / "session_report.md").exists() and ooda_records:
            last = ooda_records[-1]
            events.append({
                "type": "session_complete",
                "session_id": session_id,
                "objective_achieved": last.get("action_type") == "complete",
                "session_summary": _s(last.get("session_summary")),
            })

        events.append({"type": "stream_done"})
        return events

    # ─────────────────────────────────────────────────────────────────────
    # Vault API
    # ─────────────────────────────────────────────────────────────────────

    @app.get("/api/vault")
    async def get_vault():
        from arrai.memory.vault import VaultManager
        vault = VaultManager(vault_dir)
        entries = vault.load_entries()
        return [e.to_dict() for e in entries]

    @app.get("/api/vault/{entry_id}")
    async def get_vault_entry(entry_id: str):
        from arrai.memory.vault import VaultManager
        vault = VaultManager(vault_dir)
        entries = vault.load_entries()
        entry = next((e for e in entries if e.id == entry_id), None)
        if not entry:
            raise HTTPException(status_code=404, detail="Vault entry not found")
        content = vault.read_file(entry.vault_path)
        return {**entry.to_dict(), "content": content}

    # ─────────────────────────────────────────────────────────────────
    # API Keys  (in-memory / process-scoped only — never written to disk)
    # ─────────────────────────────────────────────────────────────────

    # Canonical env-var names that the GUI knows about
    _KNOWN_KEYS = {
        "openai":     "OPENAI_API_KEY",
        "anthropic":  "ANTHROPIC_API_KEY",
    }

    @app.get("/api/keys/status")
    async def get_key_status():
        """Return which API keys are currently set (booleans only — never the values)."""
        import os
        status = {name: bool(os.environ.get(env)) for name, env in _KNOWN_KEYS.items()}
        # Surface any extra keys that were set via POST /api/keys
        status["extra_keys"] = [
            k for k in os.environ
            if k not in _KNOWN_KEYS.values() and k.startswith("_ARRAI_EXTRA_")
        ]
        return status

    @app.post("/api/keys")
    async def set_keys(req: SetKeysRequest):
        """
        Write API keys into os.environ for this server process.

        Keys are NEVER written to disk.  They are lost when the server restarts.
        Only non-empty values are applied; passing an empty string leaves the
        existing env-var unchanged.
        """
        import os
        applied: list[str] = []

        if req.openai_api_key.strip():
            os.environ["OPENAI_API_KEY"] = req.openai_api_key.strip()
            applied.append("OPENAI_API_KEY")

        if req.anthropic_api_key.strip():
            os.environ["ANTHROPIC_API_KEY"] = req.anthropic_api_key.strip()
            applied.append("ANTHROPIC_API_KEY")

        if req.extra_key_name.strip() and req.extra_key_value.strip():
            name = req.extra_key_name.strip().upper().replace(" ", "_")
            os.environ[name] = req.extra_key_value.strip()
            applied.append(name)

        return {"applied": applied, "message": f"{len(applied)} key(s) set for this server session."}

    return app

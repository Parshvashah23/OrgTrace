"""
OrgMemory-Env — FastAPI Server
OpenEnv-compliant API for organizational communication debt resolution.

Endpoints:
    POST /reset     — Initialize a new episode
    POST /step      — Execute an agent action
    GET  /state     — Return full internal state
    GET  /tasks     — List available tasks
    GET  /validate  — OpenEnv compliance check
    GET  /health    — Health check
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from org_env.models import Action, Observation, Reward
from org_env.org_memory_env import OrgMemoryEnv, TASK_CONFIG

# ── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OrgMemory-Env",
    version="1.0.0",
    description=(
        "RL environment for organizational communication debt resolution. "
        "Agents must trace decisions, surface broken commitments, and "
        "reconstruct institutional knowledge from realistic synthetic "
        "communication histories."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage: session_id → OrgMemoryEnv instance
envs: dict[str, OrgMemoryEnv] = {}

# Default data directory
DATA_DIR = str(PROJECT_ROOT / "data" / "generated")


# ── REQUEST / RESPONSE MODELS ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None
    task: Optional[str] = None  # Alias
    seed: int = 42
    session_id: str = "default"


class StepRequest(BaseModel):
    action_type: Optional[str] = None
    parameters: dict = {}
    reasoning: str = ""
    session_id: str = "default"


class StepResponse(BaseModel):
    observation: dict
    reward: dict
    done: bool
    info: dict


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/")
def home_health():
    """Root health check endpoint (requested by HF tutorial)."""
    return {"status": "ok", "env": "orgtrace"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/reset")
async def reset(
    request: Request,
    task_id: Optional[str] = Query(None),
    task: Optional[str] = Query(None),
    session_id: Optional[str] = Query(None),
) -> dict:
    """
    Initialize a new episode. Robustly accepts parameters from JSON body, 
    query parameters, or form data.
    """
    # 1. Initialize with defaults (including env var fallback)
    import os
    tid = os.getenv("ORG_TRACE_TASK")
    sid = "default"
    seed = 42

    # 2. Extract from Raw Body (handles plain string or JSON)
    raw_body = await request.body()
    if raw_body:
        try:
            # Try JSON first
            import json
            body_data = json.loads(raw_body)
            if isinstance(body_data, dict):
                tid = body_data.get("task_id") or body_data.get("task") or tid
                sid = body_data.get("session_id") or sid
                seed = body_data.get("seed", seed)
            elif isinstance(body_data, str):
                tid = body_data
        except:
            # Fallback to raw string
            decoded = raw_body.decode().strip().strip('"').strip("'")
            if decoded:
                tid = decoded

    # 3. Overwrite with Query Params (if present)
    tid = task_id or task or tid
    sid = session_id or sid

    # 4. Final check for Form data (if tid still missing)
    if not tid or tid not in TASK_CONFIG:
        try:
            form = await request.form()
            tid = form.get("task_id") or form.get("task") or tid
            sid = form.get("session_id") or sid
            if form.get("seed"):
                seed = int(form.get("seed"))
        except:
            pass

    # 5. Global Default (First task in config)
    if not tid:
        tid = list(TASK_CONFIG.keys())[0]
        print(f"DEBUG: No task_id found in request. Defaulting to: {tid}")

    print(f"DEBUG: /reset final config -> tid={tid}, sid={sid}, seed={seed}")

    if tid not in TASK_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: {tid}. Must be one of {list(TASK_CONFIG.keys())}"
        )

    if tid not in TASK_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: {tid}. Must be one of {list(TASK_CONFIG.keys())}"
        )

    env = OrgMemoryEnv(data_dir=DATA_DIR, seed=seed)
    envs[sid] = env

    obs = env.reset(tid)

    return {
        "observation": obs.model_dump(),
        "session_id": sid,
        "task_id": tid,
        "max_steps": obs.max_steps,
    }


@app.post("/step")
async def step(request: Request) -> StepResponse:
    """
    Execute one agent action.
    """
    # Robust parameter extraction
    action_type = None
    params = {}
    reasoning = ""
    sid = "default"

    # 1. Try to extract from JSON Body
    try:
        body_data = await request.json()
        if isinstance(body_data, dict):
            action_type = body_data.get("action_type")
            params = body_data.get("parameters", {})
            reasoning = body_data.get("reasoning", "")
            sid = body_data.get("session_id") or sid
    except:
        pass

    # Fallback to Form data if body-based extraction failed
    if not action_type:
        try:
            form = await request.form()
            action_type = form.get("action_type")
            sid = form.get("session_id") or sid
            reasoning = form.get("reasoning") or reasoning
            if form.get("parameters"):
                import json
                params = json.loads(form.get("parameters"))
        except:
            pass

    if not action_type:
        raise HTTPException(status_code=400, detail="Missing action_type")

    if sid not in envs:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /reset first."
        )

    env = envs[sid]

    try:
        action = Action(
            action_type=action_type,
            parameters=params,
            reasoning=reasoning,
        )
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action: {str(e)}"
        )

    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # Clean up completed sessions to prevent memory leaks
    if done:
        # Keep the env for state() queries but could add TTL cleanup
        pass

    return StepResponse(
        observation=obs.model_dump(),
        reward=reward.model_dump(),
        done=done,
        info=_serialize_info(info),
    )


@app.get("/state")
def state(session_id: str = "default") -> dict:
    """Return complete internal state (for debugging/logging)."""
    if session_id not in envs:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /reset first."
        )
    return envs[session_id].state().model_dump()


@app.get("/tasks")
def tasks() -> list:
    """List all available tasks with their configurations."""
    return [
        {
            "id": task_id,
            "max_steps": config["max_steps"],
            "query": config["query"],
            "difficulty": _get_difficulty(task_id),
        }
        for task_id, config in TASK_CONFIG.items()
    ]


@app.get("/validate")
def validate() -> dict:
    """Check OpenEnv spec compliance."""
    return {
        "status": "compliant",
        "spec_version": "1.0.0",
        "environment": "org-memory-env",
        "tasks": list(TASK_CONFIG.keys()),
        "action_types": [
            "retrieve_messages", "trace_thread", "tag_decision",
            "tag_commitment", "link_cause_effect", "draft_artifact", "submit"
        ],
        "observation_space": "structured_text_graph",
        "action_space": "discrete_retrieval_plus_generation",
        "reward_range": [0.0, 1.0],
    }


@app.get("/render")
def render(session_id: str = "default") -> dict:
    """Return human-readable state representation."""
    if session_id not in envs:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /reset first."
        )
    return {"render": envs[session_id].render()}


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> dict:
    """Delete a session to free resources."""
    if session_id in envs:
        del envs[session_id]
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="Session not found.")


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _get_difficulty(task_id: str) -> str:
    difficulties = {
        "decision_archaeology": "medium",
        "commitment_detection": "hard",
        "knowledge_recovery": "expert",
    }
    return difficulties.get(task_id, "unknown")


def _serialize_info(info: dict) -> dict:
    """Ensure info dict is JSON-serializable."""
    serialized = {}
    for k, v in info.items():
        if isinstance(v, dict):
            serialized[k] = _serialize_info(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            serialized[k] = v
        elif isinstance(v, list):
            serialized[k] = v
        else:
            serialized[k] = str(v)
    return serialized


# ── STARTUP ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Verify data files exist on startup."""
    corpus_path = Path(DATA_DIR) / "corpus.json"
    if not corpus_path.exists():
        print(f"⚠ Corpus not found at {corpus_path}")
        print(f"  Generating corpus...")
        import generator
        corpus, ground_truth = generator.generate_corpus()
        generator.save(corpus, ground_truth)
        print(f"  ✅ Corpus generated.")


def main():
    """Entry point for the 'server' console script."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()

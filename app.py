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

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.models import Action, Observation, Reward
from env.org_memory_env import OrgMemoryEnv, TASK_CONFIG

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
    task_id: str
    seed: int = 42
    session_id: str = "default"


class StepRequest(BaseModel):
    action_type: str
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
def reset(
    request: Optional[ResetRequest] = None,
    task_id: Optional[str] = Query(None),
    session_id: str = Query("default"),
) -> dict:
    """
    Initialize a new episode.

    Accepts either a JSON body (ResetRequest) or query parameters (task_id).
    """
    # Extract values from either request body or query params
    if request:
        tid = request.task_id
        sid = request.session_id
        seed = request.seed
    elif task_id:
        tid = task_id
        sid = session_id
        seed = 42  # default seed
    else:
        raise HTTPException(
            status_code=400,
            detail="Missing task_id. Provide in JSON body or as query parameter."
        )

    if tid not in TASK_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: {tid}. "
                   f"Must be one of {list(TASK_CONFIG.keys())}"
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
def step(request: StepRequest) -> StepResponse:
    """
    Execute one agent action.

    The action must be a valid Action with action_type and parameters.
    Returns observation, reward, done flag, and info dict.
    """
    if request.session_id not in envs:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call /reset first."
        )

    env = envs[request.session_id]

    try:
        action = Action(
            action_type=request.action_type,
            parameters=request.parameters,
            reasoning=request.reasoning,
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

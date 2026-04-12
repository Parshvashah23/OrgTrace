import os
import json
import textwrap
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Any

from openai import OpenAI
from org_env.org_memory_env import OrgMemoryEnv, TASK_CONFIG
from org_env.models import Action

# ── CONFIG ───────────────────────────────────────────────────────────────────

# Task Configuration
TASK_ID = os.getenv("ORG_TRACE_TASK", "decision_archaeology")
BENCHMARK = "orgtrace"
MAX_STEPS = 20  # Default for OrgTrace tasks
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Normalize action for logging (escaped and single line)
    action_log = action.replace("\n", "\\n")
    print(
        f"[STEP] step={step} action={action_log} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── LOGIC REUSED FROM run_baseline.py ────────────────────────────────────────

ACTION_SCHEMA_JSON = json.dumps({
    "type": "object",
    "properties": {
        "action_type": {
            "type": "string",
            "enum": [
                "retrieve_messages", "trace_thread", "tag_decision",
                "tag_commitment", "link_cause_effect", "draft_artifact", "submit"
            ]
        },
        "parameters": {"type": "object"},
        "reasoning": {"type": "string"}
    },
    "required": ["action_type", "parameters", "reasoning"]
})

SYSTEM_PROMPT = """You are an organizational memory analyst. You have access to a company's
communication history. Your job is to investigate the given query, retrieve context, and submit a structured answer.

## Available Actions
You must output exactly one action per turn as valid JSON:
{action_schema}

## Strategy
1. Start broad with 'retrieve_messages'.
2. Narrow down with 'trace_thread'.
3. Build context by tagging decisions/commitments.
4. Draft artifact sections as you go.
5. Use 'submit' when done.

## Important
- Output ONLY valid JSON.
- You have {max_steps} steps total.
"""

def format_observation(obs_dict: Dict[str, Any], step: int, max_steps: int) -> str:
    parts = [f"## Step {step}/{max_steps}", f"Query: {obs_dict.get('query', '')}", ""]
    
    context = obs_dict.get("retrieved_context", [])
    if context:
        parts.append(f"## Retrieved Messages (Last 5)")
        for msg in context[-5:]:
            parts.append(f"[{msg.get('message_id')}] {msg.get('sender_name')}: {msg.get('body')[:200]}")
            
    return "\n".join(parts)


# ── MAIN LOOP ────────────────────────────────────────────────────────────────

def main():
    # State tracking variables
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    # 1. Configuration Check
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")
    
    log_start(task=TASK_ID, env=BENCHMARK, model=MODEL_NAME)

    try:
        if not HF_TOKEN:
            print("[DEBUG] Warning: HF_TOKEN environment variable is missing.", flush=True)

        # 2. Client Initialization
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "none")
        
        # 3. Environment Initialization
        # Resolve data directory relative to script
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data" / "generated"
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found at {data_dir.absolute()}")
            
        env = OrgMemoryEnv(data_dir=str(data_dir), seed=42)
        
        # 4. Episode Reset
        obs = env.reset(TASK_ID)
        max_steps = obs.max_steps
        
        system = SYSTEM_PROMPT.format(
            action_schema=ACTION_SCHEMA_JSON,
            max_steps=max_steps
        )
        
        # 5. Interaction Loop
        for step in range(1, max_steps + 1):
            obs_dict = obs.model_dump()
            user_prompt = format_observation(obs_dict, step, max_steps)
            
            # LLM Call
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                raw_action = response.choices[0].message.content.strip()
                # Simple markdown strip
                if raw_action.startswith("```"):
                    raw_action = "\n".join(raw_action.split("\n")[1:-1])
                
                action_data = json.loads(raw_action)
                action = Action(**action_data)
            except Exception as e:
                # Fallback action
                action = Action(
                    action_type="retrieve_messages",
                    parameters={"query": "update"},
                    reasoning=f"Fallback due to error: {str(e)}"
                )
                raw_action = action.model_dump_json()

            # Env Step
            obs, reward_obj, done, info = env.step(action)
            
            # Scoring & Logging
            reward = reward_obj.step_score - reward_obj.penalty
            rewards.append(reward)
            steps_taken = step
            
            log_step(step=step, action=raw_action, reward=reward, done=done, error=None)
            
            if done:
                grader_result = info.get("grader_result", {})
                score = grader_result.get("total_score", 0.0)
                success = score >= SUCCESS_SCORE_THRESHOLD
                break

    except Exception:
        print("[DEBUG] Fatal Error in Inference Script:", flush=True)
        traceback.print_exc()
    finally:
        # Ensure we always log the end for the validator
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()

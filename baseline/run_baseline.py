#!/usr/bin/env python3
"""
OrgMemory-Env — Baseline Inference Script
Runs an LLM agent against the environment using the OpenAI-compatible API.

Supports:
  - Groq (GROQ_API_KEY) — default, uses llama-3.3-70b-versatile
  - OpenAI (OPENAI_API_KEY) — fallback, uses gpt-4o-mini

Usage:
    python -m baseline.run_baseline                          # Run all 3 tasks × 3 seeds
    python -m baseline.run_baseline --task decision_archaeology --seed 42   # Single run
    python -m baseline.run_baseline --provider openai        # Force OpenAI
"""

import json
import sys
import time
import argparse
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
import os

load_dotenv()

from env.org_memory_env import OrgMemoryEnv
from env.models import Action

# ── LLM CLIENT SETUP ────────────────────────────────────────────────────────

def get_llm_client(provider: Optional[str] = None) -> Tuple[Any, str, str]:
    """
    Initialize the OpenAI-compatible LLM client.

    Returns:
        Tuple of (client, model_name, provider_name)
    """
    from openai import OpenAI

    groq_key = os.environ.get("GROQ_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if provider == "groq" or (provider is None and groq_key):
        if not groq_key:
            raise ValueError("GROQ_API_KEY not set")
        client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
        return client, "llama-3.3-70b-versatile", "groq"

    elif provider == "openai" or (provider is None and openai_key):
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = OpenAI(api_key=openai_key)
        return client, "gpt-4o-mini", "openai"

    else:
        raise ValueError(
            "No LLM API key found. Set GROQ_API_KEY or OPENAI_API_KEY in your .env file."
        )


# ── ACTION SCHEMA ────────────────────────────────────────────────────────────

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
}, indent=2)

# ── SUBMISSION SCHEMAS ───────────────────────────────────────────────────────

SUBMISSION_SCHEMAS = {
    "decision_archaeology": {
        "root_decision": "message_id of the root cause message",
        "decision_chain": ["msg_id_1", "msg_id_2", "...", "msg_id_N"],
        "accountable_person": "person_id of accountable party",
        "decision_text": "summary of the decision"
    },
    "commitment_detection": {
        "dropped_commitments": [
            {
                "source_message_id": "message_id",
                "committer": "person_id",
                "commitment_text": "what was committed",
                "risk_level": "low|medium|high|critical",
                "resolution_plan": "suggested next step"
            }
        ]
    },
    "knowledge_recovery": {
        "artifact": {
            "systems": ["system_name_1", "system_name_2"],
            "decisions": [{"decision_text": "...", "owner": "person_id"}],
            "collaborators": ["person_id_1", "person_id_2"],
            "timeline": [{"day": 1, "event": "description"}],
            "open_items": ["item_1", "item_2"],
            "freeform_notes": "additional notes"
        }
    },
}

# ── SYSTEM PROMPT ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an organizational memory analyst. You have access to a company's
communication history spanning 60 days across email, Slack, meeting notes,
and document comments.

Your job is to investigate the given query by retrieving relevant messages,
building up context incrementally, and ultimately submitting a structured answer.

## Available Actions
You must output exactly one action per turn as valid JSON matching this schema:
{action_schema}

## Action Guide
- retrieve_messages: Search by keyword, person, date, project, or channel.
  Parameters: {{"query": "search terms", "person_id": "P01" (optional), "project_id": "atlas" (optional), "top_k": 10 (optional)}}
- trace_thread: Follow a conversation thread.
  Parameters: {{"thread_id": "thread_xxx"}}
- tag_decision: Mark a message as a decision point.
  Parameters: {{"message_id": "msg_xxx", "decision_text": "what was decided"}}
- tag_commitment: Mark a message as containing a commitment.
  Parameters: {{"message_id": "msg_xxx", "committer_id": "P01", "commitment_text": "what was committed"}}
- link_cause_effect: Connect two messages causally.
  Parameters: {{"cause_message_id": "msg_xxx", "effect_message_id": "msg_yyy"}}
- draft_artifact: Write a section of your answer document.
  Parameters: {{"section": "section_name", "content": "section content"}}
- submit: Finalize and submit your answer (triggers scoring).
  Parameters: {{"answer": <submission_object>}}

## Submission Format for This Task
When you are ready to submit, use this exact structure for the "answer" field:
{submission_schema}

## Strategy
1. Start broad — retrieve messages to understand the landscape
2. Narrow down — use filters and thread tracing to find key moments
3. Build incrementally — tag decisions and commitments as you find them
4. Draft as you go — don't leave artifact writing to the last step
5. Submit when confident — you have {max_steps} steps total

## Important
- Always include reasoning in the 'reasoning' field.
- Think step by step. Be specific in retrieval queries.
- Output ONLY valid JSON. No markdown, no explanation outside the JSON.
- You MUST submit before running out of steps.
- Submit 2 steps before the step limit to avoid timeout.
"""


# ── OBSERVATION FORMATTER ────────────────────────────────────────────────────

def format_observation(obs_dict: Dict[str, Any], step: int, max_steps: int) -> str:
    """Format an observation into a concise string for the LLM."""
    parts = []

    parts.append(f"## Step {step}/{max_steps}")
    parts.append(f"Task: {obs_dict.get('task_id', 'unknown')}")
    parts.append(f"Query: {obs_dict.get('query', '')}")
    parts.append("")

    # Recently retrieved context (last batch)
    context = obs_dict.get("retrieved_context", [])
    if context:
        # Show last 10 messages
        recent = context[-10:]
        parts.append(f"## Retrieved Messages ({len(context)} total, showing last {len(recent)})")
        for msg in recent:
            sender = msg.get("sender_name", msg.get("sender_id", "?"))
            day = msg.get("day", "?")
            channel = msg.get("channel", "?")
            subject = msg.get("subject", "")
            body = msg.get("body", "")[:300]
            msg_id = msg.get("message_id", "?")
            thread_id = msg.get("thread_id", "?")
            project = msg.get("project_tag", "")

            parts.append(f"---")
            parts.append(f"[{msg_id}] Day {day} | {sender} via {channel}" +
                        (f" | Project: {project}" if project else "") +
                        (f" | Thread: {thread_id}" if thread_id else ""))
            if subject:
                parts.append(f"Subject: {subject}")
            parts.append(body)
        parts.append("")

    # Visible seed messages (only on first step)
    if step <= 1:
        visible = obs_dict.get("visible_messages", [])
        if visible:
            parts.append(f"## Seed Messages ({len(visible)} messages)")
            for msg in visible[:10]:  # Show first 10
                sender = msg.get("sender_name", "?")
                day = msg.get("day", "?")
                body = msg.get("body", "")[:200]
                msg_id = msg.get("message_id", "?")
                thread_id = msg.get("thread_id", "?")
                parts.append(f"[{msg_id}] Day {day} | {sender} | Thread: {thread_id}")
                parts.append(body[:200])
                parts.append("")

    # Org graph summary (first step only)
    if step <= 1:
        org = obs_dict.get("org_graph", {})
        teams = org.get("teams", {})
        if teams:
            parts.append("## Organization")
            for team, members in teams.items():
                people_info = org.get("people", {})
                member_names = [people_info.get(m, {}).get("name", m) for m in members[:5]]
                parts.append(f"- {team}: {', '.join(member_names)}" +
                           (f" (+{len(members)-5} more)" if len(members) > 5 else ""))
            parts.append("")

    # Project states
    if step <= 1:
        projects = obs_dict.get("project_states", [])
        if projects:
            parts.append("## Projects")
            for proj in projects:
                parts.append(f"- {proj.get('name', '?')} ({proj.get('project_id', '?')}): "
                           f"Status={proj.get('status', '?')}, "
                           f"Owner={proj.get('owner_id', '?')}, "
                           f"Lead Eng={proj.get('lead_eng_id', '?')}")
            parts.append("")

    return "\n".join(parts)


def format_reward_feedback(reward_dict: Dict[str, Any]) -> str:
    """Format reward feedback for the LLM."""
    parts = []
    feedback = reward_dict.get("feedback", "")
    if feedback:
        parts.append(f"Feedback: {feedback}")
    step_score = reward_dict.get("step_score", 0)
    penalty = reward_dict.get("penalty", 0)
    if step_score > 0 or penalty > 0:
        parts.append(f"Step score: +{step_score:.3f}, Penalty: -{penalty:.3f}")
    return " | ".join(parts) if parts else ""


# ── EPISODE RUNNER ───────────────────────────────────────────────────────────

def run_episode(
    env: OrgMemoryEnv,
    task_id: str,
    client: Any,
    model: str,
    seed: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run a single episode of the environment.

    Returns:
        Dict with task_id, seed, total_reward, final_grader_score, steps_used, component_scores
    """
    obs = env.reset(task_id)
    obs_dict = obs.model_dump()

    # Build system prompt
    submission_schema = json.dumps(SUBMISSION_SCHEMAS.get(task_id, {}), indent=2)
    system = SYSTEM_PROMPT.format(
        action_schema=ACTION_SCHEMA_JSON,
        submission_schema=submission_schema,
        max_steps=obs.max_steps,
    )

    conversation_history = []
    total_reward = 0.0
    step = 0

    while True:
        step += 1

        # Format observation
        user_content = format_observation(obs_dict, step, obs.max_steps)

        # Add reward feedback from previous step
        if step > 1 and conversation_history:
            reward_info = conversation_history[-1].get("_reward_feedback", "")
            if reward_info:
                user_content = f"[Previous action feedback: {reward_info}]\n\n{user_content}"

        # Force submit if near step limit
        if step >= obs.max_steps - 1:
            user_content += (
                "\n\n⚠️ WARNING: You are at your step limit! "
                "You MUST submit now with whatever information you have gathered. "
                "Use action_type 'submit'."
            )

        conversation_history.append({"role": "user", "content": user_content})

        # Call LLM
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system},
                    *[{"role": m["role"], "content": m["content"]}
                      for m in conversation_history[-10:]],  # Keep last 10 turns
                ],
            )
            raw_action = response.choices[0].message.content.strip()
        except Exception as e:
            if verbose:
                print(f"    ⚠ LLM error: {e}")
            # Fallback action
            raw_action = json.dumps({
                "action_type": "retrieve_messages",
                "parameters": {"query": "project update decision"},
                "reasoning": "Fallback due to API error"
            })

        conversation_history.append({"role": "assistant", "content": raw_action})

        # Parse action
        try:
            # Clean up potential markdown wrapping
            action_text = raw_action
            if action_text.startswith("```"):
                lines = action_text.split("\n")
                # Remove first and last lines (``` markers)
                lines = [l for l in lines if not l.strip().startswith("```")]
                action_text = "\n".join(lines)

            action_data = json.loads(action_text)
            action = Action(
                action_type=action_data["action_type"],
                parameters=action_data.get("parameters", {}),
                reasoning=action_data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            if verbose:
                print(f"    ⚠ Parse error (step {step}): {e}")
                print(f"      Raw: {raw_action[:200]}")

            # If near end, force submit with empty answer
            if step >= obs.max_steps - 1:
                action = Action(
                    action_type="submit",
                    parameters={"answer": SUBMISSION_SCHEMAS.get(task_id, {})},
                    reasoning="Forced submit due to parse error at step limit",
                )
            else:
                action = Action(
                    action_type="retrieve_messages",
                    parameters={"query": "decision commitment update"},
                    reasoning="Fallback due to parse error",
                )

        if verbose:
            reasoning = action.reasoning[:80] if action.reasoning else ""
            print(f"    Step {step:2d}: {action.action_type:20s} | {reasoning}")

        # Execute action
        obs, reward, done, info = env.step(action)
        obs_dict = obs.model_dump()
        reward_dict = reward.model_dump()

        total_reward += reward.step_score - reward.penalty

        # Store reward feedback for next turn
        conversation_history[-1]["_reward_feedback"] = format_reward_feedback(reward_dict)

        if verbose and reward.feedback:
            print(f"           → {reward.feedback[:100]}")

        if done:
            grader_result = info.get("grader_result", {})
            final_score = grader_result.get("total_score", 0.0) if grader_result else 0.0
            component_scores = grader_result.get("component_scores", {}) if grader_result else {}

            if verbose:
                print(f"    ✅ Done! Final score: {final_score:.3f}")
                if component_scores:
                    for k, v in component_scores.items():
                        print(f"       {k}: {v:.3f}")

            return {
                "task_id": task_id,
                "seed": seed,
                "total_reward": round(total_reward, 4),
                "final_grader_score": round(final_score, 4),
                "steps_used": step,
                "component_scores": component_scores,
            }


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run baseline agent against OrgMemory-Env"
    )
    parser.add_argument(
        "--task", type=str, default=None,
        choices=["decision_archaeology", "commitment_detection", "knowledge_recovery"],
        help="Run a single task (default: all 3)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Run with a single seed (default: 42, 123, 999)"
    )
    parser.add_argument(
        "--provider", type=str, default=None, choices=["groq", "openai"],
        help="Force LLM provider (default: auto-detect from env)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Data directory (default: data/generated/)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress step-by-step output"
    )

    args = parser.parse_args()

    # Setup tasks and seeds
    tasks = [args.task] if args.task else [
        "decision_archaeology", "commitment_detection", "knowledge_recovery"
    ]
    seeds = [args.seed] if args.seed else [42, 123, 999]

    # Setup LLM client
    try:
        client, model, provider = get_llm_client(args.provider)
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    print(f"🧠 OrgMemory-Env Baseline")
    print(f"   Provider: {provider} | Model: {model}")
    print(f"   Tasks: {', '.join(tasks)}")
    print(f"   Seeds: {seeds}")
    print(f"   Total episodes: {len(tasks) * len(seeds)}")
    print()

    # Determine data directory
    data_dir = args.data_dir or str(PROJECT_ROOT / "data" / "generated")

    # Check if corpus exists
    corpus_path = Path(data_dir) / "corpus.json"
    if not corpus_path.exists():
        print(f"⚠ Corpus not found at {corpus_path}")
        print(f"  Generating corpus...")
        sys.path.insert(0, str(PROJECT_ROOT))
        import generator
        corpus, ground_truth = generator.generate_corpus()
        generator.save(corpus, ground_truth)
        print()

    # Run episodes
    results = []
    for task_id in tasks:
        for seed in seeds:
            print(f"{'─' * 60}")
            print(f"▶ Task: {task_id} | Seed: {seed}")
            print(f"{'─' * 60}")

            env = OrgMemoryEnv(data_dir=data_dir, seed=seed)
            start_time = time.time()

            try:
                result = run_episode(
                    env, task_id, client, model, seed,
                    verbose=not args.quiet,
                )
                result["elapsed_seconds"] = round(time.time() - start_time, 1)
                result["provider"] = provider
                result["model"] = model
                results.append(result)
            except Exception as e:
                print(f"    ❌ Episode failed: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "task_id": task_id,
                    "seed": seed,
                    "total_reward": 0.0,
                    "final_grader_score": 0.0,
                    "steps_used": 0,
                    "component_scores": {},
                    "elapsed_seconds": round(time.time() - start_time, 1),
                    "provider": provider,
                    "model": model,
                    "error": str(e),
                })
            print()

    # ── RESULTS ──────────────────────────────────────────────────────────────

    print(f"\n{'═' * 60}")
    print(f"  BASELINE RESULTS")
    print(f"  Provider: {provider} | Model: {model}")
    print(f"{'═' * 60}\n")

    # Print table
    header = f"{'Task':<28s} {'Seed':>6s} {'Score':>8s} {'Steps':>7s} {'Time':>7s}"
    print(header)
    print("─" * len(header))

    for r in results:
        print(
            f"{r['task_id']:<28s} "
            f"{r['seed']:>6d} "
            f"{r['final_grader_score']:>8.3f} "
            f"{r['steps_used']:>7d} "
            f"{r.get('elapsed_seconds', 0):>6.1f}s"
        )

    print("─" * len(header))

    # Aggregate by task
    print(f"\n{'Task':<28s} {'Mean':>8s} {'Std':>8s}")
    print("─" * 46)

    for task_id in tasks:
        task_results = [r for r in results if r["task_id"] == task_id and "error" not in r]
        if task_results:
            scores = [r["final_grader_score"] for r in task_results]
            mean_score = sum(scores) / len(scores)
            if len(scores) > 1:
                variance = sum((s - mean_score) ** 2 for s in scores) / (len(scores) - 1)
                std_score = variance ** 0.5
            else:
                std_score = 0.0
            print(f"{task_id:<28s} {mean_score:>8.3f} {std_score:>8.3f}")

    # Save to CSV
    output_dir = PROJECT_ROOT / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"

    fieldnames = [
        "task_id", "seed", "final_grader_score", "total_reward",
        "steps_used", "elapsed_seconds", "provider", "model"
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📊 Results saved to {csv_path}")

    # Save detailed results as JSON
    json_path = output_dir / "results_detailed.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"📊 Detailed results saved to {json_path}")


if __name__ == "__main__":
    main()

"""
inference.py — Baseline inference script for SQL Repair Environment.
=======================================================================

MANDATORY ENVIRONMENT VARIABLES:
  API_BASE_URL    The API endpoint for the LLM (default: HF router)
  MODEL_NAME      The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN        Your Hugging Face API key
  ENV_BASE_URL    The SQL Repair environment URL (default: http://localhost:7860)

STDOUT FORMAT (strictly followed):
  [START] task=<task_name> env=sql_repair_env model=<model_name>
  [STEP]  step=<n> action=<sql_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import os
import sys
import json
import httpx
from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────────
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "sql_repair_env"
MAX_STEPS    = 5
TEMPERATURE  = 0.2

TASKS = ["syntax_fix", "logic_fix", "schema_rewrite"]

# ── OpenAI client ──────────────────────────────────────────────────────────────
client = OpenAI(api_key=API_KEY or "hf_placeholder", base_url=API_BASE_URL)
http   = httpx.Client(base_url=ENV_BASE_URL, timeout=60.0)


# ── Helpers ────────────────────────────────────────────────────────────────────

def env_reset(task_name: str) -> dict:
    r = http.post("/reset", json={"task_name": task_name})
    r.raise_for_status()
    return r.json()

def env_step(sql: str) -> dict:
    r = http.post("/step", json={"sql": sql, "reasoning": None})
    r.raise_for_status()
    return r.json()

def build_prompt(obs: dict) -> str:
    parts = [
        "You are an expert SQL engineer. Your job is to write or fix a SQL query.",
        "",
        f"TASK: {obs['task_description']}",
        "",
        "DATABASE SCHEMA:",
        obs['schema_ddl'],
        "",
    ]
    if obs.get("broken_sql"):
        parts += [
            "BROKEN QUERY (fix this):",
            obs["broken_sql"],
            "",
        ]
    parts += [
        "SAMPLE DATA (for context):",
        obs.get("sample_data", "{}"),
        "",
    ]
    if obs.get("last_error"):
        parts += [
            f"YOUR LAST ATTEMPT FAILED WITH ERROR: {obs['last_error']}",
            "",
        ]
    if obs.get("last_result_preview"):
        parts += [
            f"YOUR LAST ATTEMPT RETURNED (first 3 rows): {obs['last_result_preview']}",
            "This is WRONG — fix the query.",
            "",
        ]
    parts += [
        "Respond with ONLY the corrected SQL query. No markdown, no explanation, no backticks.",
        "Just the raw SQL.",
    ]
    return "\n".join(parts)

def call_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=512,
    )
    sql = response.choices[0].message.content.strip()
    # Strip any accidental markdown fences
    if sql.startswith("```"):
        lines = sql.split("\n")
        sql = "\n".join(l for l in lines if not l.startswith("```")).strip()
    return sql

def sanitize_for_log(text: str) -> str:
    """Remove newlines for single-line log format."""
    return text.replace("\n", " ").replace("\r", " ").strip()[:200]


# ── Episode runner ──────────────────────────────────────────────────────────────

def run_episode(task_name: str) -> dict:
    obs = env_reset(task_name)
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    rewards = []
    step = 0
    done = False
    score = 0.0
    success = False

    while not done and step < MAX_STEPS:
        step += 1
        prompt = build_prompt(obs)

        try:
            sql = call_llm(prompt)
        except Exception as e:
            sql = "SELECT 1"  # fallback on LLM error
            print(f"[STEP] step={step} action=LLM_ERROR reward=0.00 done=false error={str(e)[:100]}", flush=True)
            rewards.append(0.0)
            continue

        try:
            obs = env_step(sql)
        except Exception as e:
            print(f"[STEP] step={step} action={sanitize_for_log(sql)} reward=0.00 done=true error={str(e)[:100]}", flush=True)
            rewards.append(0.0)
            break

        reward = obs.get("reward", 0.0)
        done   = obs.get("done", False)
        error  = obs.get("last_error") or "null"
        rewards.append(reward)

        print(
            f"[STEP] step={step} "
            f"action={sanitize_for_log(sql)} "
            f"reward={reward:.2f} "
            f"done={'true' if done else 'false'} "
            f"error={sanitize_for_log(str(error)) if error != 'null' else 'null'}",
            flush=True,
        )

        if reward >= 0.6:
            score = reward
            success = True

    # Final score = best reward seen
    if rewards:
        score = max(rewards)
        success = score >= 0.6

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step} "
        f"score={score:.2f} "
        f"rewards={rewards_str}",
        flush=True,
    )

    return {"task": task_name, "score": score, "success": success, "steps": step}


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    # Check env is alive
    try:
        r = http.get("/health")
        r.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach environment at {ENV_BASE_URL} — {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    for task_name in TASKS:
        result = run_episode(task_name)
        results.append(result)
        print("", flush=True)  # blank line between tasks

    # Summary to stderr (not part of scored stdout)
    print("=" * 60, file=sys.stderr)
    print("SUMMARY", file=sys.stderr)
    for r in results:
        status = "✓" if r["success"] else "✗"
        print(f"  {status} {r['task']:<20} score={r['score']:.2f}  steps={r['steps']}", file=sys.stderr)
    avg = sum(r["score"] for r in results) / len(results)
    print(f"\n  Average score: {avg:.2f}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()

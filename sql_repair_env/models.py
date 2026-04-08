"""
models.py — Type-safe Action, Observation, and State definitions
for the SQL Repair Environment.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Base types (mirrors openenv.core.env_server.types) ────────────────────────

class Action(BaseModel):
    pass

class Observation(BaseModel):
    done: bool = False
    reward: float = 0.0

class State(BaseModel):
    episode_id: str = ""
    step_count: int = 0


# ── SQL Repair Action ──────────────────────────────────────────────────────────

class SQLRepairAction(Action):
    """
    The agent's action: a SQL query string it believes is correct.
    The agent may also optionally emit a short chain-of-thought
    explanation (not scored, but logged for analysis).
    """
    sql: str = Field(..., description="The corrected SQL query produced by the agent")
    reasoning: Optional[str] = Field(
        None, description="Optional chain-of-thought reasoning (not scored)"
    )


# ── SQL Repair Observation ─────────────────────────────────────────────────────

class SQLRepairObservation(Observation):
    """
    Observation returned to the agent after each step.
    Contains everything the agent needs to understand the task and fix the query.
    """
    task_name: str = Field(..., description="Which task is active: syntax_fix | logic_fix | schema_rewrite")
    schema_ddl: str = Field(..., description="Full CREATE TABLE DDL for the database schema")
    broken_sql: str = Field(..., description="The broken/incorrect SQL query to fix (empty for schema_rewrite)")
    task_description: str = Field(..., description="Natural language description of what the query must do")
    sample_data: str = Field(..., description="JSON-encoded sample rows to help the agent understand the data")
    last_error: Optional[str] = Field(None, description="Error message from the agent's last SQL attempt, if any")
    last_result_preview: Optional[str] = Field(None, description="Preview of last query's output rows (if it ran)")
    attempt_number: int = Field(1, description="Which attempt this is (max 5 per episode)")
    max_attempts: int = Field(5, description="Maximum attempts allowed")
    done: bool = False
    reward: float = 0.0


# ── SQL Repair State ───────────────────────────────────────────────────────────

class SQLRepairState(State):
    """
    Full episode state (returned by GET /state).
    """
    task_name: str = ""
    attempt_number: int = 0
    max_attempts: int = 5
    best_score: float = 0.0
    solved: bool = False
    expected_row_count: Optional[int] = None

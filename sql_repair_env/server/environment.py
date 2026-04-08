"""
environment.py — Core SQL Repair Environment logic.

Implements reset(), step(), and state() following the OpenEnv spec.
Each episode runs against an in-memory SQLite database seeded with
the task's schema and data.  The agent submits SQL queries and receives
scored observations.  Up to MAX_ATTEMPTS per episode.
"""

import sqlite3
import uuid
from typing import Optional

from models import SQLRepairAction, SQLRepairObservation, SQLRepairState
from tasks import TASKS, _run_sql


MAX_ATTEMPTS = 5
MIN_REWARD = 0.01
MAX_REWARD = 0.99
SOLVED_THRESHOLD = 0.99


class SQLRepairEnvironment:
    """
    Real-world RL environment: an agent repairs broken or missing SQL queries.

    Episode flow:
        obs = env.reset(task_name)   # start episode, get task context
        while not obs.done:
            action = agent.act(obs)  # agent produces a SQL string
            obs = env.step(action)   # env executes SQL, scores, returns obs
    """

    def __init__(self):
        self._conn: Optional[sqlite3.Connection] = None
        self._task_name: str = ""
        self._task: dict = {}
        self._episode_id: str = ""
        self._attempt: int = 0
        self._best_score: float = 0.0
        self._solved: bool = False
        self._done: bool = True   # start in done state until reset() called

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, task_name: str = "syntax_fix") -> SQLRepairObservation:
        """
        Initialise a new episode for the given task.
        Builds a fresh in-memory SQLite DB, seeds it, and returns the
        initial observation containing the full task context.
        """
        if task_name not in TASKS:
            task_name = "syntax_fix"

        self._task_name = task_name
        self._task = TASKS[task_name]
        self._episode_id = str(uuid.uuid4())[:12]
        self._attempt = 0
        self._best_score = 0.0
        self._solved = False
        self._done = False

        # Fresh in-memory SQLite for this episode
        if self._conn:
            self._conn.close()
        self._conn = sqlite3.connect(":memory:")
        self._conn.executescript(self._task["schema_ddl"])
        self._conn.executescript(self._task["seed_sql"])
        self._conn.commit()

        return SQLRepairObservation(
            task_name=task_name,
            schema_ddl=self._task["schema_ddl"],
            broken_sql=self._task["broken_sql"],
            task_description=self._task["task_description"],
            sample_data=self._task["sample_data"],
            last_error=None,
            last_result_preview=None,
            attempt_number=0,
            max_attempts=MAX_ATTEMPTS,
            done=False,
            reward=0.0,
        )

    def step(self, action: SQLRepairAction) -> SQLRepairObservation:
        """
        Execute the agent's SQL against the task database and return a
        scored observation.

        Reward design:
          - Correct answer on first attempt: 0.99
          - Correct answer on later attempts: 0.99 - 0.1*(attempt-1)  (min 0.6)
          - Partial progress: Jaccard similarity score (0.0–0.99)
          - Penalty per attempt: -0.05 (encourages efficiency)
          - Episode ends when: solved OR attempts exhausted
        """
        if self._done:
            # Return terminal observation if already done
            return SQLRepairObservation(
                task_name=self._task_name,
                schema_ddl=self._task.get("schema_ddl", ""),
                broken_sql=self._task.get("broken_sql", ""),
                task_description=self._task.get("task_description", ""),
                sample_data=self._task.get("sample_data", "{}"),
                last_error="Episode is already done. Call reset() to start a new episode.",
                attempt_number=self._attempt,
                max_attempts=MAX_ATTEMPTS,
                done=True,
                reward=0.0,
            )

        self._attempt += 1
        agent_sql = action.sql.strip()

        # Run agent's SQL
        agent_rows, run_error = _run_sql(self._conn, agent_sql)

        # Score it
        grader = self._task["grader"]
        raw_score = grader(agent_sql, self._conn)

        # Reward shaping
        attempt_penalty = 0.05 * (self._attempt - 1)
        if raw_score >= SOLVED_THRESHOLD:
            reward = max(MAX_REWARD - 0.1 * (self._attempt - 1), 0.6)
            self._solved = True
            self._done = True
        else:
            reward = max(raw_score - attempt_penalty, MIN_REWARD)
            if self._attempt >= MAX_ATTEMPTS:
                self._done = True

        self._best_score = max(self._best_score, raw_score)

        # Build result preview
        preview = None
        if agent_rows is not None:
            preview = str(agent_rows[:3])  # show up to 3 rows

        return SQLRepairObservation(
            task_name=self._task_name,
            schema_ddl=self._task["schema_ddl"],
            broken_sql=self._task["broken_sql"],
            task_description=self._task["task_description"],
            sample_data=self._task["sample_data"],
            last_error=run_error,
            last_result_preview=preview,
            attempt_number=self._attempt,
            max_attempts=MAX_ATTEMPTS,
            done=self._done,
            reward=round(min(max(reward, MIN_REWARD), MAX_REWARD), 4),
        )

    @property
    def state(self) -> SQLRepairState:
        """Return current episode metadata."""
        return SQLRepairState(
            episode_id=self._episode_id,
            step_count=self._attempt,
            task_name=self._task_name,
            attempt_number=self._attempt,
            max_attempts=MAX_ATTEMPTS,
            best_score=self._best_score,
            solved=self._solved,
        )

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

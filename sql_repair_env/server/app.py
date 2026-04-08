"""
app.py — FastAPI server exposing the OpenEnv HTTP API.

Endpoints:
  POST /reset          - start new episode  (body: {"task_name": "..."})
  POST /step           - execute action     (body: SQLRepairAction JSON)
  GET  /state          - get episode state
  GET  /health         - liveness probe
  GET  /tasks          - list available tasks
"""

import sys
import os

# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from models import SQLRepairAction, SQLRepairObservation, SQLRepairState
from server.environment import SQLRepairEnvironment

app = FastAPI(
    title="SQL Repair Environment",
    description="OpenEnv-compatible RL environment for SQL query repair tasks.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = SQLRepairEnvironment()


class ResetRequest(BaseModel):
    task_name: Optional[str] = "syntax_fix"


@app.get("/health")
def health():
    return {"status": "ok", "service": "sql_repair_env"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {
                "name": "syntax_fix",
                "difficulty": "easy",
                "description": "Fix SQL syntax errors (typos, missing commas, wrong keywords)",
            },
            {
                "name": "logic_fix",
                "difficulty": "medium",
                "description": "Fix a query that runs but returns wrong results",
            },
            {
                "name": "schema_rewrite",
                "difficulty": "hard",
                "description": "Write a correct multi-table query from a natural language spec",
            },
        ]
    }


@app.post("/reset", response_model=SQLRepairObservation)
def reset(req: ResetRequest = None):
    task_name = (req.task_name if req else None) or "syntax_fix"
    obs = env.reset(task_name=task_name)
    return obs


@app.post("/step", response_model=SQLRepairObservation)
def step(action: SQLRepairAction):
    obs = env.step(action)
    return obs


@app.get("/state", response_model=SQLRepairState)
def state():
    return env.state


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()

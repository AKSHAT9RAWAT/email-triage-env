# 🛠️ SQL Repair Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible RL environment where an AI agent learns to **repair and write SQL queries**.

Given a SQLite schema and a broken or missing query, the agent must produce correct SQL — graded deterministically by comparing output against the expected result set.

---

## Motivation

SQL query repair is a task humans do constantly: debugging broken queries, fixing logic errors in data pipelines, writing queries from natural language specs. This environment trains agents to do it reliably across increasing levels of difficulty — from syntax typos to multi-table rewrites.

---

## Tasks

| Task | Difficulty | Description |
|------|-----------|-------------|
| `syntax_fix` | 🟢 Easy | Fix syntax errors: typos (`SELEC` → `SELECT`), missing commas, wrong keywords (`FORM` → `FROM`) |
| `logic_fix` | 🟡 Medium | Fix logical errors: wrong JOIN type, summing the wrong column, missing WHERE clause |
| `schema_rewrite` | 🔴 Hard | Write a correct 4-table query from scratch given only a natural language spec and schema |

---

## Action Space

```python
class SQLRepairAction(BaseModel):
    sql: str           # The corrected SQL query
    reasoning: str     # Optional chain-of-thought (not scored)
```

## Observation Space

```python
class SQLRepairObservation(BaseModel):
    task_name: str              # "syntax_fix" | "logic_fix" | "schema_rewrite"
    schema_ddl: str             # Full CREATE TABLE DDL
    broken_sql: str             # The broken query to fix (empty for schema_rewrite)
    task_description: str       # Natural language spec
    sample_data: str            # JSON sample rows for context
    last_error: Optional[str]   # SQL error from last attempt
    last_result_preview: str    # First 3 rows from last attempt
    attempt_number: int         # Current attempt (max 5)
    max_attempts: int           # Always 5
    done: bool
    reward: float               # 0.0 – 1.0
```

---

## Reward Function

The reward is shaped to provide signal throughout the trajectory:

| Outcome | Reward |
|---------|--------|
| Exact match on attempt 1 | **1.0** |
| Exact match on attempt 2 | **0.9** |
| Exact match on attempt 3 | **0.8** |
| Exact match on attempt 4 | **0.7** |
| Exact match on attempt 5 | **0.6** |
| Partial match (Jaccard similarity) | **0.0 – 0.99** (minus attempt penalty) |
| SQL error / wrong result | **0.0 – ~0.3** |
| All attempts exhausted | Episode ends |

Partial credit uses Jaccard similarity between agent output rows and expected rows, minus `0.05 × (attempt - 1)` efficiency penalty.

---

## Episode Boundaries

- `reset(task_name)` → starts a fresh episode with a new in-memory SQLite DB
- `step(action)` → executes SQL, scores it, returns observation
- Episode ends when: agent produces correct output **OR** 5 attempts exhausted
- Each episode gets a **fresh isolated SQLite database** — no state leaks between episodes

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/tasks` | List available tasks |
| `POST` | `/reset` | Start new episode `{"task_name": "..."}` |
| `POST` | `/step` | Submit SQL `{"sql": "...", "reasoning": "..."}` |
| `GET` | `/state` | Get current episode state |

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct` via HF Router:

| Task | Score | Notes |
|------|-------|-------|
| `syntax_fix` | ~0.90 | Model reliably fixes simple syntax errors |
| `logic_fix` | ~0.65 | Model often misses the status filter |
| `schema_rewrite` | ~0.55 | Struggles with COALESCE + multi-join structure |

---

## Setup & Usage

### Local (dev)

```bash
# Install deps
pip install -r requirements.txt

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run baseline inference
export HF_TOKEN=your_token
python inference.py
```

### Docker

```bash
docker build -t sql-repair-env .
docker run -p 7860:7860 sql-repair-env

# In another terminal
export HF_TOKEN=your_token
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

### Python client

```python
import httpx

# Reset to start easy task
r = httpx.post("http://localhost:7860/reset", json={"task_name": "syntax_fix"})
obs = r.json()
print(obs["broken_sql"])

# Submit a fix
r = httpx.post("http://localhost:7860/step", json={
    "sql": "SELECT name, dept, salary FROM employees WHERE dept = 'Engineering' ORDER BY salary DESC",
    "reasoning": "Fixed SELEC->SELECT, added commas, fixed FORM->FROM"
})
result = r.json()
print(result["reward"])  # 1.0 if correct
```

---

## Environment Variables for inference.py

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Your Hugging Face / API key |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment server URL |

---

## Project Structure

```
sql_repair_env/
├── __init__.py           # Package exports
├── models.py             # Pydantic Action / Observation / State
├── tasks.py              # Task definitions + graders
├── client.py             # Python client (sync + async)
├── openenv.yaml          # OpenEnv manifest
├── inference.py          # Baseline inference script
├── requirements.txt
├── Dockerfile
├── README.md
└── server/
    ├── __init__.py
    ├── app.py            # FastAPI server
    └── environment.py    # Core environment logic
```

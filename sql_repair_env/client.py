"""
client.py — Python client for the SQL Repair Environment server.

Usage:
    import asyncio, httpx
    from client import SQLRepairClient

    async def main():
        async with SQLRepairClient("http://localhost:7860") as client:
            obs = await client.reset("syntax_fix")
            print(obs.task_description)
            result = await client.step("SELECT name, dept, salary FROM employees WHERE dept='Engineering' ORDER BY salary DESC")
            print(result.reward, result.done)

    asyncio.run(main())
"""

import httpx
from typing import Optional
from models import SQLRepairAction, SQLRepairObservation, SQLRepairState


class SQLRepairClient:
    """Async HTTP client for the SQL Repair Environment."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
        return self

    async def __aexit__(self, *_):
        if self._client:
            await self._client.aclose()

    async def reset(self, task_name: str = "syntax_fix") -> SQLRepairObservation:
        resp = await self._client.post("/reset", json={"task_name": task_name})
        resp.raise_for_status()
        return SQLRepairObservation(**resp.json())

    async def step(self, sql: str, reasoning: Optional[str] = None) -> SQLRepairObservation:
        action = SQLRepairAction(sql=sql, reasoning=reasoning)
        resp = await self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        return SQLRepairObservation(**resp.json())

    async def state(self) -> SQLRepairState:
        resp = await self._client.get("/state")
        resp.raise_for_status()
        return SQLRepairState(**resp.json())

    async def health(self) -> dict:
        resp = await self._client.get("/health")
        resp.raise_for_status()
        return resp.json()


# ── Sync convenience wrapper ───────────────────────────────────────────────────

class SQLRepairClientSync:
    """Synchronous HTTP client (uses httpx sync)."""

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=30.0)

    def reset(self, task_name: str = "syntax_fix") -> SQLRepairObservation:
        resp = self._client.post("/reset", json={"task_name": task_name})
        resp.raise_for_status()
        return SQLRepairObservation(**resp.json())

    def step(self, sql: str, reasoning: Optional[str] = None) -> SQLRepairObservation:
        action = SQLRepairAction(sql=sql, reasoning=reasoning)
        resp = self._client.post("/step", json=action.model_dump())
        resp.raise_for_status()
        return SQLRepairObservation(**resp.json())

    def state(self) -> SQLRepairState:
        resp = self._client.get("/state")
        resp.raise_for_status()
        return SQLRepairState(**resp.json())

    def close(self):
        self._client.close()

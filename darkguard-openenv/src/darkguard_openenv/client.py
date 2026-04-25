"""Remote client for the DarkGuard OpenEnv FastAPI service."""

from __future__ import annotations

from typing import Any

import httpx
import requests


class DarkGuardClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def reset(self, **payload: Any) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def step(self, action: dict[str, Any] | str) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/step", json={"action": action}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def state(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/state", timeout=self.timeout)
        response.raise_for_status()
        return response.json()


class AsyncDarkGuardClient:
    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def reset(self, **payload: Any) -> dict[str, Any]:
        response = await self._client.post(f"{self.base_url}/reset", json=payload)
        response.raise_for_status()
        return response.json()

    async def step(self, action: dict[str, Any] | str) -> dict[str, Any]:
        response = await self._client.post(f"{self.base_url}/step", json={"action": action})
        response.raise_for_status()
        return response.json()

    async def state(self) -> dict[str, Any]:
        response = await self._client.get(f"{self.base_url}/state")
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()

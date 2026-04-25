"""HTTP client for remote DarkGuard OpenEnv Space."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import requests
import time


@dataclass(slots=True)
class RemoteEnvClient:
    base_url: str
    timeout_s: float = 45.0
    hf_token: str | None = None
    max_retries: int = 5
    min_request_interval_s: float = 0.2
    retry_backoff_s: float = 0.8
    _session: requests.Session = field(init=False, repr=False)
    _last_request_ts: float = field(init=False, default=0.0, repr=False)

    def __post_init__(self) -> None:
        self._session = requests.Session()
        self._last_request_ts = 0.0

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        return headers

    def health(self) -> dict[str, Any]:
        response = self._request("GET", f"{self.base_url.rstrip('/')}/health", endpoint="health")
        return response.json()

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._request(
            "POST",
            f"{self.base_url.rstrip('/')}/reset",
            json=payload,
            endpoint="reset",
        )
        return response.json()

    def step(self, action: dict[str, Any] | str) -> dict[str, Any]:
        response = self._request(
            "POST",
            f"{self.base_url.rstrip('/')}/step",
            json={"action": action},
            endpoint="step",
        )
        return response.json()

    def state(self) -> dict[str, Any]:
        response = self._request("GET", f"{self.base_url.rstrip('/')}/state", endpoint="state")
        return response.json()

    def _request(self, method: str, url: str, *, endpoint: str, json: dict[str, Any] | None = None) -> requests.Response:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._respect_rate_limit_pacing()
            response = self._session.request(
                method=method,
                url=url,
                json=json,
                timeout=self.timeout_s,
                headers=self._headers(),
            )
            if response.status_code not in {429, 502, 503, 504}:
                self._raise_for_status(response, endpoint)
                return response
            wait_s = self._retry_wait_seconds(response, attempt)
            last_err = requests.HTTPError(f"{endpoint} transient failure: {response.status_code}, retry in {wait_s:.2f}s")
            time.sleep(wait_s)
        if last_err:
            raise last_err
        raise RuntimeError(f"{endpoint} failed without response")

    def _respect_rate_limit_pacing(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_ts
        wait = self.min_request_interval_s - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    def _retry_wait_seconds(self, response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.2, float(retry_after))
            except ValueError:
                pass
        return min(12.0, self.retry_backoff_s * (2**attempt))

    @staticmethod
    def _raise_for_status(response: requests.Response, endpoint: str) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            body = response.text[:500].replace("\n", " ")
            raise requests.HTTPError(
                f"{endpoint} failed: {response.status_code} {response.reason}; body={body}"
            ) from exc

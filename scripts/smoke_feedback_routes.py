from __future__ import annotations

import sys
from typing import List

import requests


BASE_URL = "http://localhost:8000"


def check(path: str, expected_status: int = 200, headers: dict | None = None) -> str:
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, timeout=15, headers=headers or {})
    if resp.status_code != expected_status:
        raise RuntimeError(f"{path} -> expected {expected_status}, got {resp.status_code}")
    return f"{path} -> {resp.status_code}"


def main() -> None:
    checks: List[str] = [
        "/feedback",
        "/feedback/preview?reviewer_id=quintin",
        "/api/feedback/predictions",
        "/api/feedback/reviewers/quintin/stats",
        "/api/feedback/reviewers/quintin/preferences",
    ]
    for path in checks:
        print(check(path))
    # Admin-gated: unauthenticated access must be rejected.
    print(check("/api/feedback/debug/predictions", expected_status=403))
    print("Smoke checks passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Smoke checks failed: {exc}")
        sys.exit(1)

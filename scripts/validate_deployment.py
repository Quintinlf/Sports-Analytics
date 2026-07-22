"""Post-deploy validation for Render or local environments.

Usage:
  FEEDBACK_BASE_URL=https://your-service.onrender.com python scripts/validate_deployment.py
  TEST_REVIEWER_ID=quintin python scripts/validate_deployment.py
"""
from __future__ import annotations

import os
import sys
from typing import List
from urllib.parse import urljoin

import requests

BASE_URL = os.getenv("FEEDBACK_BASE_URL", "").rstrip("/")
REVIEWER_ID = os.getenv("TEST_REVIEWER_ID", "quintin")
FAVORITE_SPORT = os.getenv("TEST_FAVORITE_SPORT", "MLB")


def check(path: str, expected_status: int = 200, headers: dict | None = None) -> str:
    url = urljoin(BASE_URL + "/", path.lstrip("/"))
    resp = requests.get(url, timeout=20, headers=headers or {})
    if resp.status_code != expected_status:
        raise RuntimeError(f"{path} -> expected {expected_status}, got {resp.status_code}")
    return f"{path} -> {resp.status_code}"


def main() -> None:
    if not BASE_URL:
        print("ERROR: FEEDBACK_BASE_URL is required (e.g. https://your-service.onrender.com)")
        sys.exit(1)

    checks: List[str] = [
        "/feedback",
        f"/feedback/preview?reviewer_id={REVIEWER_ID}",
        "/api/feedback/predictions",
        f"/api/feedback/reviewers/{REVIEWER_ID}/stats",
        f"/api/feedback/reviewers/{REVIEWER_ID}/preferences",
        f"/feedback?reviewer_id={REVIEWER_ID}&sport={FAVORITE_SPORT}",
    ]

    for path in checks:
        print(check(path))

    # Debug endpoint is admin-gated: unauthenticated must be 403.
    print(check("/api/feedback/debug/predictions", expected_status=403))
    admin_key = os.getenv("ADMIN_API_KEY", "").strip()
    if admin_key:
        print(
            check(
                "/api/feedback/debug/predictions",
                headers={"X-Admin-Key": admin_key},
            )
        )

    stats_url = urljoin(BASE_URL + "/", f"api/feedback/reviewers/{REVIEWER_ID}/stats")
    stats = requests.get(stats_url, timeout=20).json()
    if not stats.get("name"):
        raise RuntimeError("Reviewer stats response missing name field")

    preds = requests.get(urljoin(BASE_URL + "/", "api/feedback/predictions"), timeout=20).json()
    if not isinstance(preds, list):
        raise RuntimeError("Predictions API did not return a JSON array")

    email_link = f"{BASE_URL}/feedback?reviewer_id={REVIEWER_ID}&sport={FAVORITE_SPORT}"
    print(f"Email CTA link shape OK: {email_link}")
    print("Deployment validation passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Deployment validation failed: {exc}")
        sys.exit(1)

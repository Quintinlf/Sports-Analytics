"""CORS allowlist is env-driven and keeps localhost for local development."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch


class TestCorsOrigins(unittest.TestCase):
    def test_explicit_cors_allowed_origins(self) -> None:
        with patch.dict(
            os.environ,
            {
                "CORS_ALLOWED_ORIGINS": "https://app.example.com, http://localhost:3000",
                "FEEDBACK_BASE_URL": "https://ignored.example.com",
            },
            clear=False,
        ):
            from backend.config import get_cors_origins

            self.assertEqual(
                get_cors_origins(),
                ["https://app.example.com", "http://localhost:3000"],
            )

    def test_defaults_include_localhost_and_feedback_base(self) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("CORS_ALLOWED_ORIGINS", "FEEDBACK_BASE_URL")}
        env["FEEDBACK_BASE_URL"] = "https://sports-analytics.onrender.com"
        with patch.dict(os.environ, env, clear=True):
            # Re-import not required — function reads env at call time.
            from backend.config import get_cors_origins

            origins = get_cors_origins()
            self.assertIn("http://localhost:8000", origins)
            self.assertIn("http://127.0.0.1:8000", origins)
            self.assertIn("https://sports-analytics.onrender.com", origins)

    def test_empty_env_still_allows_localhost(self) -> None:
        env = {k: v for k, v in os.environ.items() if k not in ("CORS_ALLOWED_ORIGINS", "FEEDBACK_BASE_URL")}
        with patch.dict(os.environ, env, clear=True):
            from backend.config import get_cors_origins

            origins = get_cors_origins()
            self.assertEqual(
                origins,
                ["http://localhost:8000", "http://127.0.0.1:8000"],
            )


if __name__ == "__main__":
    unittest.main()

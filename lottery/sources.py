"""Fetching draw history from the New York State open-data portal.

New York publishes every Powerball and Mega Millions drawing as a public
Socrata dataset. No API key is needed for reasonable volumes; set
``SOCRATA_APP_TOKEN`` in the environment to raise the rate limit.

This is the whole ingest surface. It returns raw rows and does no
interpretation, so parsing bugs stay in ``history.py`` where they are tested.
"""
from __future__ import annotations

import os
from typing import Iterator, Optional

import requests

from lottery.games import Game

__all__ = ["SOCRATA_HOST", "fetch_raw", "fetch_draws", "LotterySourceError"]

SOCRATA_HOST = "https://data.ny.gov"

#: Socrata caps a single page; we page until a short page comes back.
PAGE_SIZE = 5000

DEFAULT_TIMEOUT = 30


class LotterySourceError(RuntimeError):
    """The upstream dataset could not be read."""


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json"}
    token = os.getenv("SOCRATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token
    return headers


def fetch_raw(
    game: Game,
    *,
    since: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    session: Optional[requests.Session] = None,
) -> Iterator[dict]:
    """Yield every published row for ``game``, oldest first.

    ``since`` is an ISO date; when given, only later drawings are fetched,
    which is what the incremental refresh uses.
    """
    url = f"{SOCRATA_HOST}/resource/{game.resource_id}.json"
    http = session or requests.Session()
    offset = 0

    while True:
        params = {
            "$limit": PAGE_SIZE,
            "$offset": offset,
            "$order": "draw_date ASC",
        }
        if since:
            params["$where"] = f"draw_date > '{since}T00:00:00'"

        try:
            response = http.get(
                url, params=params, headers=_headers(), timeout=timeout
            )
            response.raise_for_status()
            page = response.json()
        except requests.RequestException as exc:
            raise LotterySourceError(
                f"Failed fetching {game.name} from {url}: {exc}"
            ) from exc
        except ValueError as exc:
            raise LotterySourceError(
                f"{game.name} endpoint returned non-JSON content"
            ) from exc

        if not isinstance(page, list):
            raise LotterySourceError(
                f"{game.name} endpoint returned {type(page).__name__}, expected a list"
            )

        yield from page

        if len(page) < PAGE_SIZE:
            return
        offset += PAGE_SIZE


def fetch_draws(game: Game, *, since: Optional[str] = None, **kwargs):
    """Fetch and parse in one step. Returns ``list[Draw]`` oldest first."""
    from lottery.history import parse_records

    return parse_records(game, fetch_raw(game, since=since, **kwargs))

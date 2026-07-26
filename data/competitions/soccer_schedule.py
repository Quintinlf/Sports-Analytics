"""TheSportsDB soccer schedule provider — multi-league discovery."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from data.competitions.resolver import resolve_active
from data.competitions.types import Competition, UnifiedFixture

logger = logging.getLogger(__name__)

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

DEFAULT_API_BASE = "https://www.thesportsdb.com/api/v1/json/3"
# Cap per league so club calendars don't drown international fixtures.
_EVENTS_PER_LEAGUE = 8


class SoccerScheduleProvider:
    """Fetch and merge upcoming fixtures across catalogued soccer competitions."""

    def __init__(self, api_base: str = DEFAULT_API_BASE, session: Any = None):
        self.api_base = api_base.rstrip("/")
        self._session = session

    def fetch_fixtures(self) -> List[UnifiedFixture]:
        """Fetch eventsnextleague for active soccer competitions; merge + dedupe."""
        if not REQUESTS_AVAILABLE and self._session is None:
            logger.warning("requests not available; soccer schedule empty")
            return []

        competitions = resolve_active("SOCCER")
        if not competitions:
            logger.warning("No soccer competitions in catalog")
            return []

        merged: List[UnifiedFixture] = []
        for comp in competitions:
            if not comp.provider_league_id:
                logger.warning(
                    "Missing provider_league_id for competition %s (%s)",
                    comp.id,
                    comp.display_name,
                )
                continue
            events = self._fetch_league_events(comp)
            for ev in events:
                fixture = self._event_to_fixture(ev, comp)
                if fixture is not None:
                    merged.append(fixture)

        return self._dedupe(merged)

    def fetch_as_fifa_dicts(self) -> List[Dict[str, Any]]:
        """Convenience: UnifiedFixture → FIFALivePredictionService shape."""
        return [f.to_fifa_dict() for f in self.fetch_fixtures()]

    def _http_get(self, url: str) -> Optional[Any]:
        getter = self._session.get if self._session is not None else requests.get
        try:
            resp = getter(url, timeout=15)
        except Exception as exc:
            logger.error("Soccer API request failed for %s: %s", url, exc)
            return None
        if getattr(resp, "status_code", 200) != 200:
            logger.error(
                "Soccer API failure status=%s url=%s",
                getattr(resp, "status_code", "?"),
                url,
            )
            return None
        try:
            return resp.json()
        except Exception as exc:
            logger.error("Soccer API JSON parse failed for %s: %s", url, exc)
            return None

    def _fetch_league_events(self, comp: Competition) -> List[Dict[str, Any]]:
        url = f"{self.api_base}/eventsnextleague.php?id={comp.provider_league_id}"
        payload = self._http_get(url)
        if payload is None:
            return []

        events = payload.get("events")
        if events is None:
            logger.info(
                "Empty soccer response for %s (league_id=%s)",
                comp.display_name,
                comp.provider_league_id,
            )
            return []
        if not events:
            logger.info(
                "No upcoming events for %s (league_id=%s)",
                comp.display_name,
                comp.provider_league_id,
            )
            return []

        return list(events)[:_EVENTS_PER_LEAGUE]

    def _event_to_fixture(
        self, ev: Dict[str, Any], comp: Competition
    ) -> Optional[UnifiedFixture]:
        provider_game_id = str(ev.get("idEvent") or "").strip()
        home = (ev.get("strHomeTeam") or "").strip()
        away = (ev.get("strAwayTeam") or "").strip()
        if not provider_game_id or not home or not away:
            logger.warning(
                "Skipping incomplete event in %s: id=%r home=%r away=%r",
                comp.display_name,
                ev.get("idEvent"),
                ev.get("strHomeTeam"),
                ev.get("strAwayTeam"),
            )
            return None

        league_name = (
            (ev.get("strLeague") or "").strip()
            or comp.display_name
        )
        game_date = str(ev.get("dateEvent") or "").strip() or ""

        return UnifiedFixture(
            sport="SOCCER",
            competition_id=comp.id,
            league=league_name,
            provider_game_id=provider_game_id,
            game_date=game_date,
            home_team=home,
            away_team=away,
            predict_policy=comp.predict_policy,
            meta={"provider_league_id": comp.provider_league_id},
        )

    @staticmethod
    def _dedupe(fixtures: List[UnifiedFixture]) -> List[UnifiedFixture]:
        """Keep first occurrence of each provider_game_id (catalog priority order)."""
        seen: set[str] = set()
        out: List[UnifiedFixture] = []
        for f in fixtures:
            if f.provider_game_id in seen:
                continue
            seen.add(f.provider_game_id)
            out.append(f)
        return out


def fetch_soccer_fixtures(
    api_base: str = DEFAULT_API_BASE,
    session: Any = None,
) -> List[UnifiedFixture]:
    return SoccerScheduleProvider(api_base=api_base, session=session).fetch_fixtures()

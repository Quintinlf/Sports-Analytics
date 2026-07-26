"""NBA live scoreboard schedule provider — no historical FINAL fallback."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from data.competitions.types import NbaSeasonPhase, PredictPolicy, UnifiedFixture

logger = logging.getLogger(__name__)

# nba_api live scoreboard: 1=scheduled, 2=in progress, 3=final
_STATUS_FINAL = 3


class NbaScheduleProvider:
    """Discover upcoming NBA games from the live scoreboard only."""

    def __init__(self, scoreboard_fetcher=None):
        """Optional *scoreboard_fetcher* returns a list of raw game dicts (tests)."""
        self._scoreboard_fetcher = scoreboard_fetcher

    def fetch_fixtures(self) -> List[UnifiedFixture]:
        raw_games, phase = self._load_scoreboard()
        upcoming = [g for g in raw_games if self._is_upcoming(g)]

        if not upcoming:
            logger.info(
                "NBA offseason or empty scoreboard (phase=%s): no upcoming games",
                phase.value,
            )
            return []

        competition_id = (
            "nba_playoffs" if phase == NbaSeasonPhase.PLAYOFFS else "nba_regular"
        )
        league = (
            "NBA Playoffs" if phase == NbaSeasonPhase.PLAYOFFS else "NBA"
        )

        fixtures: List[UnifiedFixture] = []
        for g in upcoming:
            game_id = str(g.get("game_id") or g.get("GAME_ID") or "").strip()
            home = (g.get("home_team") or "").strip()
            away = (g.get("away_team") or "").strip()
            if not game_id or not home or not away:
                continue
            game_date = str(g.get("game_date") or "")[:10]
            fixtures.append(
                UnifiedFixture(
                    sport="NBA",
                    competition_id=competition_id,
                    league=league,
                    provider_game_id=game_id,
                    game_date=game_date,
                    home_team=home,
                    away_team=away,
                    predict_policy=PredictPolicy.FULL_MODEL,
                    season_phase=phase.value,
                    meta={
                        "GAME_ID": game_id,
                        "game_status": g.get("game_status"),
                        "game_status_text": g.get("game_status_text"),
                        "home_team_id": g.get("home_team_id"),
                        "away_team_id": g.get("away_team_id"),
                    },
                )
            )

        logger.info(
            "NBA schedule: %d upcoming fixture(s) (phase=%s)",
            len(fixtures),
            phase.value,
        )
        return fixtures

    def fetch_as_nba_dicts(self) -> List[Dict[str, Any]]:
        return [f.to_nba_dict() for f in self.fetch_fixtures()]

    def detect_season_phase(self, raw_games: Optional[List[Dict[str, Any]]] = None) -> NbaSeasonPhase:
        if raw_games is None:
            raw_games, phase = self._load_scoreboard()
            if not raw_games:
                return NbaSeasonPhase.OFFSEASON
            return phase
        upcoming = [g for g in raw_games if self._is_upcoming(g)]
        if not upcoming:
            return NbaSeasonPhase.OFFSEASON
        return self._infer_phase(upcoming)

    def _load_scoreboard(self) -> Tuple[List[Dict[str, Any]], NbaSeasonPhase]:
        if self._scoreboard_fetcher is not None:
            raw = list(self._scoreboard_fetcher() or [])
        else:
            raw = self._fetch_live_scoreboard()

        if not raw:
            return [], NbaSeasonPhase.OFFSEASON

        upcoming = [g for g in raw if self._is_upcoming(g)]
        if not upcoming:
            return raw, NbaSeasonPhase.OFFSEASON
        return raw, self._infer_phase(upcoming)

    @staticmethod
    def _fetch_live_scoreboard() -> List[Dict[str, Any]]:
        try:
            from data.nba_loader import fetch_upcoming_games

            return list(fetch_upcoming_games(verbose=False) or [])
        except Exception as exc:
            logger.error("NBA scoreboard fetch failed: %s", exc, exc_info=True)
            return []

    @staticmethod
    def _is_upcoming(game: Dict[str, Any]) -> bool:
        """Exclude completed (FINAL) games — never surface as upcoming."""
        status = game.get("game_status")
        text = str(game.get("game_status_text") or "").strip().upper()
        if status is not None:
            try:
                if int(status) == _STATUS_FINAL:
                    return False
            except (TypeError, ValueError):
                pass
        if text in {"FINAL", "FINAL/OT", "FINAL/2OT", "F"} or text.startswith("FINAL"):
            return False
        return True

    @staticmethod
    def _infer_phase(games: List[Dict[str, Any]]) -> NbaSeasonPhase:
        """Playoff game IDs typically start with 004; regular season with 002."""
        for g in games:
            gid = str(g.get("game_id") or g.get("GAME_ID") or "")
            if gid.startswith("004"):
                return NbaSeasonPhase.PLAYOFFS
        return NbaSeasonPhase.REGULAR


def fetch_nba_fixtures(scoreboard_fetcher=None) -> List[UnifiedFixture]:
    return NbaScheduleProvider(scoreboard_fetcher=scoreboard_fetcher).fetch_fixtures()

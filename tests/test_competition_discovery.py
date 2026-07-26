"""Year-round competition discovery: catalog, soccer merge/dedupe, NBA offseason."""
from __future__ import annotations

import json
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock

from data.competitions.catalog import (
    REQUIRED_SOCCER_LEAGUE_IDS,
    all_competitions,
    competitions_for_sport,
)
from data.competitions.nba_schedule import NbaScheduleProvider
from data.competitions.soccer_schedule import SoccerScheduleProvider
from data.competitions.types import PredictPolicy
from data.fifa_predictions_service import FIFALivePredictionService, SCHEDULE_ONLY_NOTE


class _FakeResp:
    def __init__(self, payload: Dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Dict[str, Any]:
        return self._payload


class TestCompetitionCatalog(unittest.TestCase):
    def test_catalog_contains_required_soccer_competitions(self) -> None:
        soccer = competitions_for_sport("SOCCER")
        ids = {c.provider_league_id for c in soccer if c.provider_league_id}
        self.assertTrue(REQUIRED_SOCCER_LEAGUE_IDS.issubset(ids))

        by_id = {c.provider_league_id: c for c in soccer}
        self.assertEqual(by_id["4429"].predict_policy, PredictPolicy.FULL_MODEL)
        self.assertEqual(by_id["4502"].predict_policy, PredictPolicy.FULL_MODEL)
        for club_id in ("4328", "4335", "4331", "4332", "4334", "4480"):
            self.assertEqual(
                by_id[club_id].predict_policy,
                PredictPolicy.SCHEDULE_ONLY,
                msg=f"club league {club_id} should be schedule_only",
            )

    def test_catalog_has_nba_entries(self) -> None:
        nba = competitions_for_sport("NBA")
        self.assertGreaterEqual(len(nba), 2)
        self.assertTrue(any(c.id == "nba_regular" for c in nba))
        self.assertTrue(any(c.id == "nba_offseason" for c in all_competitions()))


class TestSoccerScheduleProvider(unittest.TestCase):
    def test_merges_leagues_and_dedupes_by_provider_game_id(self) -> None:
        session = MagicMock()

        def fake_get(url: str, timeout: int = 15):
            if "id=4328" in url:
                return _FakeResp(
                    {
                        "events": [
                            {
                                "idEvent": "dup-1",
                                "strHomeTeam": "Arsenal",
                                "strAwayTeam": "Chelsea",
                                "dateEvent": "2026-08-15",
                                "strLeague": "English Premier League",
                            },
                            {
                                "idEvent": "epl-2",
                                "strHomeTeam": "Liverpool",
                                "strAwayTeam": "Everton",
                                "dateEvent": "2026-08-16",
                                "strLeague": "English Premier League",
                            },
                        ]
                    }
                )
            if "id=4429" in url:
                return _FakeResp(
                    {
                        "events": [
                            {
                                "idEvent": "dup-1",  # duplicate across leagues
                                "strHomeTeam": "Arsenal",
                                "strAwayTeam": "Chelsea",
                                "dateEvent": "2026-08-15",
                                "strLeague": "FIFA World Cup",
                            },
                            {
                                "idEvent": "wc-1",
                                "strHomeTeam": "France",
                                "strAwayTeam": "Brazil",
                                "dateEvent": "2026-06-20",
                                "strLeague": "FIFA World Cup",
                            },
                        ]
                    }
                )
            # Other leagues: empty
            return _FakeResp({"events": None})

        session.get.side_effect = fake_get
        provider = SoccerScheduleProvider(session=session)
        fixtures = provider.fetch_fixtures()
        ids = [f.provider_game_id for f in fixtures]

        self.assertIn("epl-2", ids)
        self.assertIn("wc-1", ids)
        self.assertEqual(ids.count("dup-1"), 1)
        self.assertEqual(len(ids), len(set(ids)))

        self.assertTrue(any(f.league for f in fixtures))
        self.assertTrue(any(f.home_team == "France" for f in fixtures))
        epl = next(f for f in fixtures if f.provider_game_id == "epl-2")
        self.assertEqual(epl.predict_policy, PredictPolicy.SCHEDULE_ONLY)
        wc = next(f for f in fixtures if f.provider_game_id == "wc-1")
        self.assertEqual(wc.predict_policy, PredictPolicy.FULL_MODEL)


class TestNbaScheduleProvider(unittest.TestCase):
    def test_empty_scoreboard_returns_empty_list(self) -> None:
        provider = NbaScheduleProvider(scoreboard_fetcher=lambda: [])
        self.assertEqual(provider.fetch_fixtures(), [])
        self.assertEqual(provider.fetch_as_nba_dicts(), [])

    def test_never_returns_final_games_as_upcoming(self) -> None:
        raw = [
            {
                "game_id": "0022500001",
                "game_date": "2026-01-01",
                "home_team": "Lakers",
                "away_team": "Celtics",
                "game_status": 3,
                "game_status_text": "Final",
            },
            {
                "game_id": "0022500002",
                "game_date": "2026-01-02",
                "home_team": "Heat",
                "away_team": "Nets",
                "game_status": 1,
                "game_status_text": "7:00 pm ET",
            },
        ]
        provider = NbaScheduleProvider(scoreboard_fetcher=lambda: raw)
        fixtures = provider.fetch_fixtures()
        self.assertEqual(len(fixtures), 1)
        self.assertEqual(fixtures[0].provider_game_id, "0022500002")
        self.assertNotEqual(fixtures[0].meta.get("game_status"), 3)

    def test_all_final_scoreboard_is_offseason_empty(self) -> None:
        raw = [
            {
                "game_id": "0022500099",
                "game_date": "2026-01-01",
                "home_team": "Lakers",
                "away_team": "Celtics",
                "game_status": 3,
                "game_status_text": "Final",
            }
        ]
        provider = NbaScheduleProvider(scoreboard_fetcher=lambda: raw)
        self.assertEqual(provider.fetch_fixtures(), [])


class _StubFifaModel:
    def predict_match(self, squad_profiles, home, away):
        # Club / unknown squads → no model score
        if home in squad_profiles and away in squad_profiles:
            return {"HOME_WIN": 0.5, "DRAW": 0.25, "AWAY_WIN": 0.25}
        return None

    def squad_metric_maps(self, squad_profiles, home, away):
        return {}, {}


class TestFifaScheduleOnly(unittest.TestCase):
    def test_schedule_only_fixture_does_not_fabricate_ai_prediction(self) -> None:
        service = FIFALivePredictionService()
        model_bundle = {
            "model": _StubFifaModel(),
            "squad_profiles": {"France": {}, "Brazil": {}},
        }
        fixtures = [
            {
                "id": "club-1",
                "league": "Premier League",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "utc_date": "2026-08-15",
                "predict_policy": PredictPolicy.SCHEDULE_ONLY.value,
            }
        ]
        rows = service.build_prediction_rows(fixtures, model_bundle)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row["is_fallback"])
        self.assertEqual(row["predicted_winner"], "Scheduled")
        self.assertEqual(row["win_probability"], 0.0)
        self.assertIsNone(row["model_name"])
        snap = json.loads(row["feature_snapshot"])
        self.assertTrue(snap["metrics"].get("schedule_only"))
        self.assertEqual(snap["metrics"].get("discovery_note"), SCHEDULE_ONLY_NOTE)
        # Must not look like a scored AI pick
        self.assertNotIn(row["predicted_winner"], ("Arsenal", "Chelsea", "Draw"))

    def test_full_model_missing_squad_becomes_schedule_only(self) -> None:
        service = FIFALivePredictionService()
        model_bundle = {
            "model": _StubFifaModel(),
            "squad_profiles": {"France": {}},
        }
        fixtures = [
            {
                "id": "wc-x",
                "league": "FIFA World Cup",
                "home_team": "France",
                "away_team": "Unknown FC",
                "utc_date": "2026-06-20",
                "predict_policy": PredictPolicy.FULL_MODEL.value,
            }
        ]
        rows = service.build_prediction_rows(fixtures, model_bundle)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["is_fallback"])
        self.assertEqual(rows[0]["predicted_winner"], "Scheduled")


if __name__ == "__main__":
    unittest.main()

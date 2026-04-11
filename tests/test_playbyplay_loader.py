import unittest
from unittest.mock import patch

import pandas as pd

from data.playbyplay_loader import (
    _game_seconds_remaining,
    _parse_pctimestring,
    _parse_score_margin,
    format_play_by_play_to_state_nodes,
    get_last_n_state_nodes,
)


class PlayByPlayLoaderTests(unittest.TestCase):
    def test_parse_helpers(self):
        self.assertEqual(_parse_score_margin("TIE"), 0)
        self.assertEqual(_parse_score_margin("-7"), -7)
        self.assertEqual(_parse_score_margin(None), 0)

        self.assertEqual(_parse_pctimestring("11:34"), 694)
        self.assertEqual(_parse_pctimestring("bad"), 0)

        # Q4 with 00:45 left means 45 game seconds remaining.
        self.assertEqual(_game_seconds_remaining(4, "00:45"), 45)
        # Q2 with 01:00 left means full Q4+Q3 and 1 min in Q2 remaining.
        self.assertEqual(_game_seconds_remaining(2, "01:00"), 1500)

    def test_format_play_by_play_to_state_nodes(self):
        pbp_df = pd.DataFrame(
            [
                {
                    "PERIOD": 1,
                    "EVENTNUM": 1,
                    "EVENTMSGTYPE": 1,
                    "EVENTMSGACTIONTYPE": 1,
                    "PCTIMESTRING": "11:45",
                    "SCORE": "2-0",
                    "SCOREMARGIN": "2",
                    "HOMEDESCRIPTION": "HOME makes 2-pt shot",
                    "VISITORDESCRIPTION": None,
                    "NEUTRALDESCRIPTION": None,
                    "PLAYER1_TEAM_ID": 100,
                },
                {
                    "PERIOD": 1,
                    "EVENTNUM": 2,
                    "EVENTMSGTYPE": 5,
                    "EVENTMSGACTIONTYPE": 1,
                    "PCTIMESTRING": "11:30",
                    "SCORE": "2-0",
                    "SCOREMARGIN": "2",
                    "HOMEDESCRIPTION": None,
                    "VISITORDESCRIPTION": "AWAY turnover",
                    "NEUTRALDESCRIPTION": None,
                    "PLAYER1_TEAM_ID": 200,
                },
                {
                    "PERIOD": 1,
                    "EVENTNUM": 3,
                    "EVENTMSGTYPE": 1,
                    "EVENTMSGACTIONTYPE": 1,
                    "PCTIMESTRING": "11:00",
                    "SCORE": "2-3",
                    "SCOREMARGIN": "-1",
                    "HOMEDESCRIPTION": None,
                    "VISITORDESCRIPTION": "AWAY makes 3PT shot",
                    "NEUTRALDESCRIPTION": None,
                    "PLAYER1_TEAM_ID": 200,
                },
            ]
        )

        nodes = format_play_by_play_to_state_nodes(
            pbp_df=pbp_df,
            game_id="0000000001",
            home_team_id=100,
            away_team_id=200,
        )

        self.assertEqual(len(nodes), 3)
        self.assertEqual(nodes.iloc[0]["action_label"], "shot_made_2pt")
        self.assertEqual(nodes.iloc[2]["action_label"], "shot_made_3pt")
        self.assertEqual(int(nodes.iloc[0]["possession_home"]), 1)
        self.assertEqual(int(nodes.iloc[1]["possession_home"]), 0)
        self.assertEqual(int(nodes.iloc[2]["score_margin"]), -1)

    @patch("data.playbyplay_loader.fetch_play_by_play")
    @patch("data.playbyplay_loader.fetch_recent_game_ids")
    def test_get_last_n_state_nodes_orchestration(self, mock_recent_games, mock_pbp):
        mock_recent_games.return_value = pd.DataFrame(
            [
                {
                    "GAME_ID": "0001",
                    "GAME_DATE": pd.Timestamp("2026-03-01"),
                    "HOME_TEAM_ID": 100,
                    "AWAY_TEAM_ID": 200,
                },
                {
                    "GAME_ID": "0002",
                    "GAME_DATE": pd.Timestamp("2026-03-03"),
                    "HOME_TEAM_ID": 300,
                    "AWAY_TEAM_ID": 100,
                },
            ]
        )

        sample_pbp = pd.DataFrame(
            [
                {
                    "PERIOD": 1,
                    "EVENTNUM": 1,
                    "EVENTMSGTYPE": 12,
                    "EVENTMSGACTIONTYPE": 0,
                    "PCTIMESTRING": "12:00",
                    "SCORE": None,
                    "SCOREMARGIN": "TIE",
                    "HOMEDESCRIPTION": None,
                    "VISITORDESCRIPTION": None,
                    "NEUTRALDESCRIPTION": "Start Period",
                    "PLAYER1_TEAM_ID": None,
                }
            ]
        )
        mock_pbp.return_value = sample_pbp

        nodes = get_last_n_state_nodes(team_id=100, n_games=2, sleep_seconds=0.0)

        self.assertEqual(len(nodes), 2)
        self.assertSetEqual(set(nodes["game_id"].astype(str).tolist()), {"0001", "0002"})
        self.assertIn("game_date", nodes.columns)


if __name__ == "__main__":
    unittest.main()

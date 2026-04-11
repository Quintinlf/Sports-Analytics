"""Play-by-play ingestion and state-node formatting utilities.

This module adds a lightweight nba_api workflow for pulling the most recent
games for a team and converting each play-by-play event into a consistent
state-node table usable by downstream game-theory feature pipelines.
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv2


def _parse_score_margin(score_margin: object) -> int:
    """Parse nba_api SCOREMARGIN into integer home margin."""
    if score_margin is None:
        return 0
    text = str(score_margin).strip().upper()
    if not text or text == "N/A" or text == "NAN" or text == "TIE":
        return 0
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return 0


def _parse_pctimestring(pctimestring: object) -> int:
    """Convert PCTIMESTRING (MM:SS) to seconds remaining in period."""
    if pctimestring is None:
        return 0
    text = str(pctimestring).strip()
    if ":" not in text:
        return 0
    minute_str, second_str = text.split(":", 1)
    try:
        minutes = int(minute_str)
        seconds = int(second_str)
    except ValueError:
        return 0
    return max(0, minutes * 60 + seconds)


def _period_length_seconds(period: int) -> int:
    return 720 if period <= 4 else 300


def _game_seconds_remaining(period: int, pctimestring: object) -> int:
    """Approximate remaining game seconds from period and game clock."""
    if period < 1:
        return 0
    secs_remaining_period = _parse_pctimestring(pctimestring)
    if period <= 4:
        future_regulation = (4 - period) * 720
        return secs_remaining_period + future_regulation
    return secs_remaining_period


def _score_margin_bucket(score_margin: int) -> int:
    if score_margin <= -15:
        return -3
    if score_margin <= -6:
        return -2
    if score_margin <= -1:
        return -1
    if score_margin == 0:
        return 0
    if score_margin <= 5:
        return 1
    if score_margin <= 14:
        return 2
    return 3


def _time_bucket(seconds_remaining_game: int) -> int:
    # Buckets align with late-game decision granularity used by state models.
    if seconds_remaining_game <= 120:
        return 1
    if seconds_remaining_game <= 240:
        return 2
    if seconds_remaining_game <= 420:
        return 3
    if seconds_remaining_game <= 720:
        return 4
    return 5


def _classify_action(event_msg_type: int, event_msg_action_type: int, row: pd.Series) -> str:
    home_desc = str(row.get("HOMEDESCRIPTION") or "")
    away_desc = str(row.get("VISITORDESCRIPTION") or "")
    neutral_desc = str(row.get("NEUTRALDESCRIPTION") or "")
    desc_upper = f"{home_desc} {away_desc} {neutral_desc}".upper()

    if event_msg_type == 1:
        return "shot_made_3pt" if "3PT" in desc_upper else "shot_made_2pt"
    if event_msg_type == 2:
        return "shot_missed_3pt" if "3PT" in desc_upper else "shot_missed_2pt"
    if event_msg_type == 3:
        return "free_throw"
    if event_msg_type == 4:
        return "rebound"
    if event_msg_type == 5:
        return "turnover"
    if event_msg_type == 6:
        return "foul"
    if event_msg_type == 7:
        return "violation"
    if event_msg_type == 8:
        return "substitution"
    if event_msg_type == 9:
        return "timeout"
    if event_msg_type == 10:
        return "jump_ball"
    if event_msg_type == 12:
        return "period_start"
    if event_msg_type == 13:
        return "period_end"
    return f"other_{event_msg_type}_{event_msg_action_type}"


def _infer_possession_home(
    row: pd.Series,
    home_team_id: int,
    away_team_id: int,
    prior_possession_home: int,
) -> int:
    event_msg_type = int(row.get("EVENTMSGTYPE") or 0)
    team_id = row.get("PLAYER1_TEAM_ID")
    actor_home: Optional[bool] = None

    if pd.notna(team_id):
        try:
            team_id_int = int(team_id)
            if team_id_int == home_team_id:
                actor_home = True
            elif team_id_int == away_team_id:
                actor_home = False
        except (TypeError, ValueError):
            actor_home = None

    if actor_home is None:
        if pd.notna(row.get("HOMEDESCRIPTION")) and str(row.get("HOMEDESCRIPTION")).strip():
            actor_home = True
        elif pd.notna(row.get("VISITORDESCRIPTION")) and str(row.get("VISITORDESCRIPTION")).strip():
            actor_home = False

    if event_msg_type in (1, 2, 3, 4, 5) and actor_home is not None:
        return int(actor_home)
    return int(prior_possession_home)


def fetch_recent_game_ids(
    team_id: int,
    n_games: int = 10,
    season: Optional[str] = None,
    season_type: str = "Regular Season",
    verbose: bool = False,
) -> pd.DataFrame:
    """Fetch the most recent unique game IDs for a team with home/away context."""
    finder = leaguegamefinder.LeagueGameFinder(
        team_id_nullable=team_id,
        season_nullable=season,
        season_type_nullable=season_type,
        league_id_nullable="00",
    )
    games = finder.get_data_frames()[0].copy()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    recent = (
        games.sort_values("GAME_DATE", ascending=False)
        .drop_duplicates(subset=["GAME_ID"]) 
        .head(int(n_games))
        .copy()
    )

    rows = []
    for _, row in recent.iterrows():
        game_id = str(row["GAME_ID"])
        game_rows = games[games["GAME_ID"] == game_id]

        home_row = game_rows[game_rows["MATCHUP"].astype(str).str.contains(r"vs\.", na=False)]
        away_row = game_rows[game_rows["MATCHUP"].astype(str).str.contains("@", na=False)]

        if not home_row.empty:
            home_team_id = int(home_row.iloc[0]["TEAM_ID"])
        elif len(game_rows) >= 2:
            home_team_id = int(game_rows.iloc[0]["TEAM_ID"])
        else:
            home_team_id = int(row["TEAM_ID"])

        if not away_row.empty:
            away_team_id = int(away_row.iloc[0]["TEAM_ID"])
        elif len(game_rows) >= 2:
            other = game_rows[game_rows["TEAM_ID"] != home_team_id]
            away_team_id = int(other.iloc[0]["TEAM_ID"]) if not other.empty else int(row["TEAM_ID"])
        else:
            away_team_id = int(row["TEAM_ID"])

        rows.append(
            {
                "GAME_ID": game_id,
                "GAME_DATE": pd.to_datetime(row["GAME_DATE"]),
                "HOME_TEAM_ID": home_team_id,
                "AWAY_TEAM_ID": away_team_id,
            }
        )

    out = pd.DataFrame(rows).sort_values("GAME_DATE", ascending=False).reset_index(drop=True)
    if verbose:
        print(f"Fetched {len(out)} recent game IDs for team {team_id}")
    return out


def fetch_play_by_play(game_id: str, timeout: int = 60) -> pd.DataFrame:
    """Fetch play-by-play rows for one game ID."""
    endpoint = playbyplayv2.PlayByPlayV2(game_id=game_id, timeout=timeout)
    pbp_df = endpoint.get_data_frames()[0].copy()
    pbp_df["GAME_ID"] = str(game_id)
    return pbp_df


def format_play_by_play_to_state_nodes(
    pbp_df: pd.DataFrame,
    game_id: str,
    home_team_id: int,
    away_team_id: int,
) -> pd.DataFrame:
    """Format raw play-by-play events into BasketballStateNode-compatible rows."""
    if pbp_df.empty:
        return pd.DataFrame(
            columns=[
                "game_id",
                "event_num",
                "period",
                "pctimestring",
                "seconds_remaining_period",
                "seconds_remaining_game",
                "score",
                "score_margin",
                "score_margin_bucket",
                "time_bucket",
                "possession_home",
                "action_label",
                "event_msg_type",
                "event_msg_action_type",
                "home_team_id",
                "away_team_id",
                "actor_team_id",
            ]
        )

    rows = pbp_df.sort_values(["PERIOD", "EVENTNUM"]).copy()
    records = []
    possession_home = 1

    for _, row in rows.iterrows():
        period = int(row.get("PERIOD") or 0)
        event_num = int(row.get("EVENTNUM") or 0)
        event_msg_type = int(row.get("EVENTMSGTYPE") or 0)
        event_msg_action_type = int(row.get("EVENTMSGACTIONTYPE") or 0)

        pctimestring = str(row.get("PCTIMESTRING") or "")
        score_margin = _parse_score_margin(row.get("SCOREMARGIN"))
        sec_remaining_period = _parse_pctimestring(pctimestring)
        sec_remaining_game = _game_seconds_remaining(period, pctimestring)
        action_label = _classify_action(event_msg_type, event_msg_action_type, row)
        possession_home = _infer_possession_home(
            row=row,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            prior_possession_home=possession_home,
        )

        actor_team_id = row.get("PLAYER1_TEAM_ID")
        actor_team_id = int(actor_team_id) if pd.notna(actor_team_id) else np.nan

        records.append(
            {
                "game_id": str(game_id),
                "event_num": event_num,
                "period": period,
                "pctimestring": pctimestring,
                "seconds_remaining_period": sec_remaining_period,
                "seconds_remaining_game": sec_remaining_game,
                "score": row.get("SCORE") if pd.notna(row.get("SCORE")) else "",
                "score_margin": score_margin,
                "score_margin_bucket": _score_margin_bucket(score_margin),
                "time_bucket": _time_bucket(sec_remaining_game),
                "possession_home": int(possession_home),
                "action_label": action_label,
                "event_msg_type": event_msg_type,
                "event_msg_action_type": event_msg_action_type,
                "home_team_id": int(home_team_id),
                "away_team_id": int(away_team_id),
                "actor_team_id": actor_team_id,
            }
        )

    return pd.DataFrame.from_records(records)


def get_last_n_state_nodes(
    team_id: int,
    n_games: int = 10,
    season: Optional[str] = None,
    season_type: str = "Regular Season",
    sleep_seconds: float = 0.5,
    verbose: bool = False,
) -> pd.DataFrame:
    """Convenience wrapper: fetch last N games and return stacked state nodes."""
    game_context_df = fetch_recent_game_ids(
        team_id=team_id,
        n_games=n_games,
        season=season,
        season_type=season_type,
        verbose=verbose,
    )

    all_nodes = []
    for _, context in game_context_df.iterrows():
        game_id = str(context["GAME_ID"])
        home_team_id = int(context["HOME_TEAM_ID"])
        away_team_id = int(context["AWAY_TEAM_ID"])

        try:
            pbp_df = fetch_play_by_play(game_id=game_id)
            nodes = format_play_by_play_to_state_nodes(
                pbp_df=pbp_df,
                game_id=game_id,
                home_team_id=home_team_id,
                away_team_id=away_team_id,
            )
            nodes["game_date"] = pd.to_datetime(context["GAME_DATE"])
            all_nodes.append(nodes)
            if verbose:
                print(f"Processed {game_id}: {len(nodes)} events")
        except Exception as exc:
            if verbose:
                print(f"Skipping {game_id}: {exc}")
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))

    if not all_nodes:
        return pd.DataFrame()

    return pd.concat(all_nodes, ignore_index=True)

"""Does the air a game is played in help predict it?

    python scripts/air_density_study.py                 # 2022-2025, cached after the first run
    python scripts/air_density_study.py --seasons 2023 2024

Pulls every regular-season MLB game with its first-pitch temperature, wind and
ballpark elevation from MLB's schedule API (a few bulk requests per season),
computes air density with data/park_physics.py, and asks three questions:

1. Do thinner-air games score more runs -- across parks, and within a park
   (where the variation is temperature, not altitude)?
2. Does air density say anything about *who wins*? Both teams bat in the same
   air, so the expectation is no.
3. Out of sample (train on the earlier seasons, test on the last): does adding
   air density and wind to each park's average improve a total-runs forecast?

Seasons start in 2022 because every park has stored balls in a humidor since
then; mixing eras would confound the ball with the air.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from data.park_physics import game_air  # noqa: E402

CACHE = REPO / "data" / "physics"
MONTHS = [("03-15", "04-30"), ("05-01", "05-31"), ("06-01", "06-30"), ("07-01", "07-31"),
          ("08-01", "08-31"), ("09-01", "10-06")]


def fetch_season(season: int) -> list[dict]:
    path = CACHE / f"mlb_games_{season}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    import statsapi

    games = []
    for start, end in MONTHS:
        r = statsapi.get("schedule", {"sportId": 1, "gameType": "R", "startDate": f"{season}-{start}",
                                      "endDate": f"{season}-{end}", "hydrate": "weather,venue(location)"})
        for day in r.get("dates", []):
            for g in day.get("games", []):
                if g.get("status", {}).get("abstractGameState") != "Final":
                    continue
                home, away = g["teams"]["home"], g["teams"]["away"]
                if home.get("score") is None or away.get("score") is None:
                    continue
                venue = g.get("venue", {})
                games.append({
                    "game_pk": g["gamePk"], "date": g.get("officialDate") or day["date"], "season": season,
                    "venue_id": venue.get("id"), "venue": venue.get("name"),
                    "elevation_ft": (venue.get("location") or {}).get("elevation"),
                    "weather": g.get("weather") or {},
                    "home_score": home["score"], "away_score": away["score"],
                })
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(games), encoding="utf-8")
    return games


def table(games: list[dict]):
    rows = []
    for g in games:
        if g["elevation_ft"] is None:
            continue
        air = game_air(g["weather"], float(g["elevation_ft"]))
        if air is None:
            continue
        rows.append((g["season"], g["venue_id"], air.density, air.temp_f, float(g["elevation_ft"]),
                     air.wind.out_component, float(air.indoors),
                     g["home_score"] + g["away_score"], float(g["home_score"] > g["away_score"])))
    a = np.array(rows, dtype=float)
    return {"season": a[:, 0], "park": a[:, 1].astype(int), "rho": a[:, 2], "temp": a[:, 3],
            "elev": a[:, 4], "wind_out": a[:, 5], "indoors": a[:, 6], "runs": a[:, 7], "home_win": a[:, 8]}


def corr(x, y) -> float:
    return float(np.corrcoef(x, y)[0, 1])


def within(values, groups):
    out = values.astype(float).copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= values[m].mean()
    return out


def perm_p(x, y, trials=2000, seed=7) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(corr(x, y))
    hits = sum(abs(corr(rng.permutation(x), y)) >= obs for _ in range(trials))
    return (hits + 1) / (trials + 1)


def design(t, mask, parks, extra):
    cols = [(t["park"][mask] == p).astype(float) for p in parks]
    cols += [t[name][mask] for name in extra]
    return np.column_stack(cols)


def forecast(t, train, test, extra):
    parks = sorted(set(t["park"][train]))
    test = test & np.isin(t["park"], parks)
    X, Xt = design(t, train, parks, extra), design(t, test, parks, extra)
    beta, *_ = np.linalg.lstsq(X, t["runs"][train], rcond=None)
    err = t["runs"][test] - Xt @ beta
    return {"mae": float(np.mean(np.abs(err))), "rmse": float(math.sqrt(np.mean(err ** 2))),
            "n_test": int(test.sum()), "coef": dict(zip(extra, map(float, beta[len(parks):])))}


def run(seasons: list[int]) -> dict:
    games = [g for s in seasons for g in fetch_season(s)]
    t = table(games)
    outdoor = t["indoors"] == 0
    rho_within = within(t["rho"], t["park"])
    runs_within = within(t["runs"], t["park"])
    last = max(seasons)
    train, test = t["season"] < last, t["season"] == last
    return {
        "seasons": seasons,
        "games": int(len(t["runs"])),
        "density_range": [float(t["rho"].min()), float(t["rho"].max())],
        "runs_vs_density": {
            "all_games_r": corr(t["rho"], t["runs"]),
            "within_park_r": corr(rho_within, runs_within),
            "within_park_p": perm_p(rho_within, runs_within),
            "outdoor_within_park_r": corr(rho_within[outdoor], runs_within[outdoor]),
        },
        "runs_vs_wind_out": {"outdoor_r": corr(t["wind_out"][outdoor], t["runs"][outdoor]),
                             "outdoor_within_park_r": corr(within(t["wind_out"], t["park"])[outdoor],
                                                           runs_within[outdoor])},
        "home_win_vs_density": {"within_park_r": corr(rho_within, within(t["home_win"], t["park"])),
                                "within_park_p": perm_p(rho_within, within(t["home_win"], t["park"]))},
        "forecast_total_runs": {
            "train_seasons": [s for s in seasons if s < last], "test_season": last,
            "park_average_only": forecast(t, train, test, []),
            "plus_air_and_wind": forecast(t, train, test, ["rho", "wind_out"]),
        },
    }


def report(res: dict) -> str:
    rv, wv, hv, fc = (res["runs_vs_density"], res["runs_vs_wind_out"], res["home_win_vs_density"],
                      res["forecast_total_runs"])
    base, plus = fc["park_average_only"], fc["plus_air_and_wind"]
    lines = [
        f"{res['games']:,} MLB games, seasons {res['seasons'][0]}-{res['seasons'][-1]} (humidor era)",
        f"air density {res['density_range'][0]:.3f}-{res['density_range'][1]:.3f} kg/m^3",
        "",
        "1. Runs vs air density (negative r = thinner air, more runs)",
        f"   across all games      r = {rv['all_games_r']:+.3f}   (mostly Coors Field vs everywhere else)",
        f"   within the same park  r = {rv['within_park_r']:+.3f}   p = {rv['within_park_p']:.4f}",
        f"   outdoor, within park  r = {rv['outdoor_within_park_r']:+.3f}",
        f"   wind blowing out      r = {wv['outdoor_within_park_r']:+.3f} (outdoor, within park)",
        "",
        "2. Home win vs air density",
        f"   within the same park  r = {hv['within_park_r']:+.3f}   p = {hv['within_park_p']:.3f}",
        "",
        f"3. Forecasting total runs in {fc['test_season']} (trained on {fc['train_seasons']})",
        f"   park average only      MAE {base['mae']:.3f}  RMSE {base['rmse']:.3f}",
        f"   + air density + wind   MAE {plus['mae']:.3f}  RMSE {plus['rmse']:.3f}"
        f"   (coef: {', '.join(f'{k} {v:+.2f}' for k, v in plus['coef'].items())})",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    args = parser.parse_args()
    res = run(sorted(args.seasons))
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "air_density_study.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(report(res))


if __name__ == "__main__":
    main()

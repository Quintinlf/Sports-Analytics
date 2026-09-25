"""The air a baseball flies through, from what MLB publishes about each game.

MLB's schedule API gives, per game, the first-pitch temperature, the wind as
"8 mph, Out To CF", whether the roof is closed, and the ballpark's elevation.
From those this module computes the air density, the quantity that actually
sets drag on a batted ball (drag = C_D * 1/2 * rho * v^2 * A).

Pressure comes from elevation through the standard-atmosphere barometric law
(the gas form of dp/dz = -rho g; Schaum's Fluid Dynamics ch. 2), not from a
game-day barometer, which MLB does not publish. Humidity is not published
either, so density is for dry air: moist air is slightly *less* dense, by up to
about 1-2% on a hot humid night -- smaller than the temperature and elevation
effects, but it means the humid-night values here are a little high.

The ball itself changed too: every MLB park has stored balls in a humidor since
2022, which is why analyses here start in 2022 (the "Schrodinger's Bat" humidor
question is about exactly this).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

R_DRY = 287.05           # J/(kg K)
SEA_LEVEL_PA = 101_325.0
INDOOR = {"dome", "roof closed"}


def station_pressure(elevation_ft: float) -> float:
    """Standard-atmosphere pressure at an elevation, Pa."""
    h = elevation_ft * 0.3048
    return SEA_LEVEL_PA * (1 - 2.25577e-5 * h) ** 5.25588


def air_density(temp_f: float, elevation_ft: float) -> float:
    """Dry-air density, kg/m^3, from temperature and elevation."""
    kelvin = (temp_f - 32) * 5 / 9 + 273.15
    return station_pressure(elevation_ft) / (R_DRY * kelvin)


SEA_LEVEL_70F = air_density(70.0, 0.0)

_WIND = re.compile(r"(\d+)\s*mph,?\s*(.*)", re.I)


@dataclass(frozen=True)
class Wind:
    mph: float
    direction: str
    out_component: float    # +mph blowing out to the outfield, -mph blowing in, 0 across or none


def parse_wind(raw: Optional[str], condition: Optional[str] = None) -> Wind:
    """'8 mph, Out To CF' -> Wind(8, 'Out To CF', +8). Indoors there is no wind."""
    if condition and condition.strip().lower() in INDOOR:
        return Wind(0.0, "indoors", 0.0)
    m = _WIND.match((raw or "").strip())
    if not m:
        return Wind(0.0, "", 0.0)
    mph, direction = float(m.group(1)), m.group(2).strip()
    d = direction.lower()
    sign = 1.0 if d.startswith("out") else -1.0 if d.startswith("in") else 0.0
    return Wind(mph, direction, sign * mph)


@dataclass(frozen=True)
class GameAir:
    temp_f: float
    elevation_ft: float
    density: float
    density_ratio: float     # vs 70 F at sea level
    indoors: bool
    wind: Wind


def game_air(weather: dict, elevation_ft: float) -> Optional[GameAir]:
    """Everything above for one game, or None when the temperature is missing."""
    try:
        temp = float(weather.get("temp"))
    except (TypeError, ValueError):
        return None
    condition = weather.get("condition") or ""
    rho = air_density(temp, elevation_ft)
    return GameAir(temp, elevation_ft, rho, rho / SEA_LEVEL_70F,
                   condition.strip().lower() in INDOOR, parse_wind(weather.get("wind"), condition))


def carry_change_estimate(density_ratio: float) -> float:
    """First-order fractional change in drag force for the same batted ball.

    Drag scales with density, so a ratio of 0.83 (Coors Field on a warm day)
    means about 17% less drag. Converting that into feet of carry needs a
    trajectory model and a drag coefficient that itself changes through the
    drag crisis (ch. 5), so this returns the drag change only.
    """
    return density_ratio - 1.0

"""Air density from published game conditions; feature checks on settled games."""
from __future__ import annotations

import unittest

from backend.feature_report import assess, to_number
from data.park_physics import SEA_LEVEL_70F, air_density, game_air, parse_wind, station_pressure


class TestParkPhysics(unittest.TestCase):
    def test_standard_pressure_and_coors_field(self) -> None:
        self.assertAlmostEqual(station_pressure(0), 101_325.0)
        coors = air_density(70, 5_190) / SEA_LEVEL_70F       # MLB lists Coors Field near 5,190 ft
        self.assertAlmostEqual(coors, 0.83, delta=0.01)

    def test_hot_air_is_thinner(self) -> None:
        self.assertLess(air_density(95, 0), air_density(55, 0))
        self.assertAlmostEqual(air_density(59, 0), 1.225, delta=0.002)   # ISA sea level, 15 C

    def test_wind_parsing(self) -> None:
        self.assertEqual(parse_wind("8 mph, Out To CF").out_component, 8)
        self.assertEqual(parse_wind("12 mph, In From LF").out_component, -12)
        self.assertEqual(parse_wind("5 mph, L To R").out_component, 0)
        self.assertEqual(parse_wind("7 mph, Out To RF", "Roof Closed").direction, "indoors")
        self.assertEqual(parse_wind(None).mph, 0)

    def test_game_air_needs_a_temperature(self) -> None:
        self.assertIsNone(game_air({"condition": "Clear"}, 20))
        air = game_air({"temp": "69", "wind": "8 mph, Out To CF", "condition": "Sunny"}, 270)
        self.assertFalse(air.indoors)
        self.assertLess(air.density_ratio, 1.0)


class TestFeatureReport(unittest.TestCase):
    def test_parsing_records_levels_and_numbers(self) -> None:
        self.assertAlmostEqual(to_number("6-4"), 0.6)
        self.assertAlmostEqual(to_number("5-3-2"), 0.65)
        self.assertEqual(to_number("moderate"), 2.0)
        self.assertEqual(to_number(True), 1.0)
        self.assertIsNone(to_number("No upcoming games"))
        self.assertIsNone(to_number(float("nan")))

    def test_leakage_is_flagged_and_small_samples_say_so(self) -> None:
        rows = []
        for i in range(40):
            home_win = i % 2 == 0
            rows.append({"sport": "NBA", "home_win": home_win, "model_correct": i % 3 == 0,
                         "features": {"home_won": float(home_win), "noise": float((i * 7) % 11)}})
        report = assess(rows, trials=200)["NBA"]
        self.assertFalse(report["enough_to_select"])
        self.assertIn("monitor", report["note"])
        by_name = {f["feature"]: f for f in report["features"]}
        self.assertTrue(by_name["home_won"]["likely_leakage"])
        self.assertFalse(by_name["noise"]["likely_leakage"])
        self.assertEqual(report["features"][0]["feature"], "home_won")


if __name__ == "__main__":
    unittest.main()

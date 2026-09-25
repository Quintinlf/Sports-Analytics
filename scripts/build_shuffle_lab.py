"""Write the shuffle laboratory's stored reports.

    python scripts/build_shuffle_lab.py

Runs ``shuffle_analysis.assess`` on every named procedure with a fixed seed and
writes ``poker/data/shuffle_lab.json``. The web lab reads that file instead of
simulating per request: a full assessment takes several seconds per procedure,
and cutting the trial count to make it fast would change the answers (the
significance test loses power, so under-mixed decks start to look fair).
Re-run after changing anything in ``poker/shuffle.py`` or
``poker/shuffle_analysis.py``; the output is deterministic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poker.shuffle import PROCEDURES  # noqa: E402
from poker.shuffle_analysis import assess  # noqa: E402

OUTPUT = REPO_ROOT / "poker" / "data" / "shuffle_lab.json"
TRIALS = 3000
SEED = 20260820


def report_dict(report) -> dict:
    test = report.position_test
    return {
        "mean_rising_sequences": round(report.mean_rising_sequences, 3),
        "order_preservation": round(report.order_preservation, 4),
        "position_test_p_value": test.p_value,
        "significant_fraction": report.significant_fraction,
        "exact_tv": report.exact_tv,
    }


def build(trials: int = TRIALS, seed: int = SEED, output: Path = OUTPUT) -> dict:
    reports = {}
    for key, procedure in PROCEDURES.items():
        started = time.perf_counter()
        reports[key] = report_dict(assess(procedure, trials=trials, seed=seed))
        print(f"  {key:<20} {time.perf_counter() - started:5.1f}s", flush=True)
    payload = {
        "generated_by": "scripts/build_shuffle_lab.py",
        "settings": {"trials": trials, "seed": seed,
                     "significance": "position chi-square at alpha 0.05, replicated"},
        "reports": reports,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    build(args.trials, args.seed)
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

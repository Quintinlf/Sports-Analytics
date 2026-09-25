"""Fail loudly if live sport model artifacts are missing.

Used by GitHub Actions before cron ingest and by production smoke.

Exit codes:
  0 — all required artifacts present
  1 — one or more missing / unreadable
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODELS_DIR = os.path.join(ROOT, "machine_learning", "models")


def _check_pointer(pointer_name: str, keys: List[str]) -> Tuple[bool, List[str]]:
    pointer_path = os.path.join(MODELS_DIR, pointer_name)
    errors: List[str] = []
    if not os.path.isfile(pointer_path):
        return False, [f"missing pointer: {pointer_path}"]
    try:
        with open(pointer_path, encoding="utf-8") as fh:
            pointer = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        return False, [f"unreadable pointer {pointer_path}: {exc}"]

    for key in keys:
        rel = pointer.get(key)
        if not rel:
            errors.append(f"{pointer_name}: missing key {key!r}")
            continue
        path = os.path.join(MODELS_DIR, os.path.basename(str(rel)))
        if not os.path.isfile(path):
            errors.append(f"missing artifact: {path}")
        elif os.path.getsize(path) < 64:
            errors.append(f"artifact too small (<64 bytes): {path}")
    return (len(errors) == 0), errors


def check_all(sports: List[str] | None = None) -> int:
    wanted = {s.upper() for s in (sports or ["NBA", "MLB", "FIFA"])}
    ok_all = True
    print("Model artifact check")
    print(f"  models dir: {MODELS_DIR}")

    if "NBA" in wanted:
        ok, errs = _check_pointer(
            "nba_latest.json",
            ["gp_path", "lgbm_win_path", "lgbm_quantile_path", "elo_path"],
        )
        if ok:
            print("[ OK ] NBA: pointer + component pickles present")
        else:
            ok_all = False
            for e in errs:
                print(f"[FAIL] NBA: {e}")

    if "MLB" in wanted:
        ok, errs = _check_pointer("mlb_latest.json", ["lgbm_win_path"])
        if ok:
            print("[ OK ] MLB: pointer + win model present")
        else:
            ok_all = False
            for e in errs:
                print(f"[FAIL] MLB: {e}")

    if "FIFA" in wanted or "SOCCER" in wanted:
        path = os.path.join(MODELS_DIR, "fifa_ensemble.pkl")
        if os.path.isfile(path) and os.path.getsize(path) >= 64:
            print("[ OK ] FIFA: fifa_ensemble.pkl present")
        else:
            ok_all = False
            print(f"[FAIL] FIFA: missing or empty {path}")

    return 0 if ok_all else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sport",
        nargs="*",
        default=None,
        help="Subset of sports (default: NBA MLB FIFA)",
    )
    args = parser.parse_args()
    return check_all(args.sport)


if __name__ == "__main__":
    raise SystemExit(main())

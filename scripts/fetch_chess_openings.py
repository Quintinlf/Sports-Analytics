"""Fetch the lichess opening names and store them with their moves in UCI.

    python -m scripts.fetch_chess_openings

Source: https://github.com/lichess-org/chess-openings (public domain / CC0). Writes
chess_coach/data/openings.tsv with columns eco, name, uci -- the PGN converted once here so the
app never parses 3,000+ PGN lines at startup.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import chess
import requests

BASE = "https://raw.githubusercontent.com/lichess-org/chess-openings/master/{}.tsv"
OUT = Path(__file__).resolve().parent.parent / "chess_coach" / "data" / "openings.tsv"


def main() -> None:
    rows = []
    for part in "abcde":
        text = requests.get(BASE.format(part), timeout=60).text
        for row in csv.DictReader(io.StringIO(text), delimiter="\t"):
            board = chess.Board()
            ucis = []
            for token in row["pgn"].split():
                if token.endswith("."):
                    continue
                ucis.append(board.push_san(token).uci())
            rows.append((row["eco"], row["name"], " ".join(ucis)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["eco", "name", "uci"])
        w.writerows(rows)
    print(f"wrote {len(rows)} openings to {OUT}")


if __name__ == "__main__":
    main()

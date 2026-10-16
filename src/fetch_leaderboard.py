#!/usr/bin/env python3
"""Snapshot the five Track 1 leaderboards to work/leaderboard.json.

Must run under work/venv-submit (kaggle>=2 needs Python 3.11+). Note page_size: the API
defaults to 20 rows, which silently hid our own entry once the field grew to 25 teams and made
it look as though we had dropped off the board entirely.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM = "Tony"
COMPS = [("CD-H", "v2d-challenge-track1-cd-h"), ("CD-O", "v2d-challenge-track1-cd-o"),
         ("ACC-H", "v2d-challenge-track1-acc-h"), ("ACC-O", "v2d-challenge-track1-acc-o"),
         ("PEN", "v2d-challenge-track1-pen")]


def main() -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    out = {"fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "metrics": {}}
    for name, slug in COMPS:
        rows = [(e.team_name, float(e.score))
                for e in api.competition_leaderboard_view(slug, page_size=200)]
        mine = next(((i + 1, v) for i, (t, v) in enumerate(rows) if t == TEAM), None)
        sc = [v for _, v in rows]
        out["metrics"][name] = dict(
            teams=len(rows), rank=mine[0] if mine else None, score=mine[1] if mine else None,
            best=sc[0], third=sc[2] if len(sc) > 2 else None,
            median=sc[len(sc) // 2], worst=sc[-1],
            top=[{"team": t, "score": v} for t, v in rows[:5]])
    ranks = [m["rank"] for m in out["metrics"].values() if m["rank"]]
    out["rank_sum"] = sum(ranks) if len(ranks) == len(COMPS) else None
    (ROOT / "work/leaderboard.json").write_text(json.dumps(out, indent=1))
    for n, m in out["metrics"].items():
        print(f"{n:<6} {m['score']:>10.5f}  rank {m['rank']}/{m['teams']}  (第1 {m['best']:.5f})")


if __name__ == "__main__":
    main()

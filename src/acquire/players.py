"""Fetch leaguedashplayerstats per season — one row per (player, team-they-played-for, season).

Enables Yu & Yang utility-network construction (player nodes with box-score
features) and other player-level analysis.
"""
import json
import sqlite3
import sys
import time

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from src import nba_patch  # noqa: F401
from nba_api.stats.endpoints import leaguedashplayerstats

from src.config import DB_PATH, REQUEST_SLEEP, REQUEST_TIMEOUT, SEASONS

MEASURE_TYPES = ("Base", "Advanced")
MAX_RETRIES = 4
BACKOFFS = [5, 15, 60, 180]


def _retry(fn, *args, **kwargs):
    last_err = None
    for i in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if isinstance(e, json.JSONDecodeError) and i == 1:
                nba_patch.reset_session()
            print(f"  retry {i+1}/{MAX_RETRIES} after {BACKOFFS[i]}s ({type(e).__name__})", flush=True)
            time.sleep(BACKOFFS[i])
    raise last_err


def fetch(season: str, measure_type: str) -> pd.DataFrame:
    def _call():
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            measure_type_detailed_defense=measure_type,
            per_mode_detailed="PerGame",
            timeout=REQUEST_TIMEOUT,
        ).get_data_frames()[0]

    df = _retry(_call)
    df.insert(0, "SEASON", season)
    df.insert(1, "MEASURE_TYPE", measure_type)
    return df


def done_seasons(conn: sqlite3.Connection, table: str) -> set:
    try:
        return {r[0] for r in conn.execute(f"SELECT DISTINCT SEASON FROM {table}")}
    except sqlite3.OperationalError:
        return set()


def run():
    conn = sqlite3.connect(DB_PATH)
    for measure in MEASURE_TYPES:
        table = f"player_stats_{measure.lower()}"
        already = done_seasons(conn, table)
        for season in SEASONS:
            if season in already:
                print(f"player_stats {season} {measure}: skip", flush=True)
                continue
            df = fetch(season, measure)
            df.to_sql(table, conn, if_exists="append", index=False)
            print(f"player_stats {season} {measure}: {len(df)} rows", flush=True)
            time.sleep(REQUEST_SLEEP)
    conn.close()


if __name__ == "__main__":
    run()

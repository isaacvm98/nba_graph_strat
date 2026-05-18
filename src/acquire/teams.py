import sqlite3
import time

import pandas as pd

from src import nba_patch  # noqa: F401  patches nba_api before import
from nba_api.stats.endpoints import leaguedashteamstats

from src.config import DB_PATH, REQUEST_SLEEP, REQUEST_TIMEOUT, SEASONS


def fetch_team_stats(season: str, measure_type: str) -> pd.DataFrame:
    df = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star="Regular Season",
        measure_type_detailed_defense=measure_type,
        per_mode_detailed="PerGame",
        timeout=REQUEST_TIMEOUT,
    ).get_data_frames()[0]
    df.insert(0, "SEASON", season)
    df.insert(1, "MEASURE_TYPE", measure_type)
    return df


def run():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    for season in SEASONS:
        for measure in ("Base", "Advanced"):
            print(f"team_stats {season} {measure}", flush=True)
            df = fetch_team_stats(season, measure)
            table = f"team_stats_{measure.lower()}"
            df.to_sql(table, conn, if_exists="append", index=False)
            time.sleep(REQUEST_SLEEP)
    conn.close()


if __name__ == "__main__":
    run()

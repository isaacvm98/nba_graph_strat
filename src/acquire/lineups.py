import sqlite3
import time

import pandas as pd

from src import nba_patch  # noqa: F401  patches nba_api before import
from nba_api.stats.endpoints import leaguedashlineups
from nba_api.stats.static import teams as static_teams

from src.config import DB_PATH, NBA_API_HEADERS, REQUEST_SLEEP, REQUEST_TIMEOUT, SEASONS

MAX_RETRIES = 7
BACKOFFS = [5, 15, 30, 60, 120, 240, 480]


def fetch_team_lineups(season: str, measure_type: str, team_id: int) -> pd.DataFrame:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            df = leaguedashlineups.LeagueDashLineups(
                season=season,
                season_type_all_star="Regular Season",
                measure_type_detailed_defense=measure_type,
                per_mode_detailed="Totals",
                group_quantity=5,
                team_id_nullable=team_id,
                timeout=REQUEST_TIMEOUT,
                headers=NBA_API_HEADERS,
            ).get_data_frames()[0]
            df.insert(0, "SEASON", season)
            df.insert(1, "MEASURE_TYPE", measure_type)
            return df
        except Exception as e:
            last_err = e
            backoff = BACKOFFS[attempt]
            print(f"  retry {attempt+1}/{MAX_RETRIES} after {backoff}s ({type(e).__name__})", flush=True)
            time.sleep(backoff)
    raise last_err


def done_teams(conn: sqlite3.Connection, table: str, season: str) -> set:
    try:
        rows = conn.execute(
            f"SELECT DISTINCT TEAM_ID FROM {table} WHERE SEASON=?", (season,)
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        return set()


def run():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    teams = static_teams.get_teams()
    conn = sqlite3.connect(DB_PATH)

    for season in SEASONS:
        for measure in ("Base", "Advanced"):
            table = f"lineup_stats_{measure.lower()}"
            already = done_teams(conn, table, season)
            todo = [t for t in teams if t["id"] not in already]
            if not todo:
                print(f"lineups {season} {measure}: skip (complete)", flush=True)
                continue
            print(f"lineups {season} {measure}: {len(todo)} teams to fetch", flush=True)
            failed = []
            for team in todo:
                try:
                    df = fetch_team_lineups(season, measure, team["id"])
                except Exception as e:
                    print(f"  SKIP {team['abbreviation']}: {type(e).__name__}", flush=True)
                    failed.append(team["abbreviation"])
                    continue
                df.to_sql(table, conn, if_exists="append", index=False)
                time.sleep(REQUEST_SLEEP)
            tag = f"  done {season} {measure}"
            if failed:
                tag += f" (failed: {failed})"
            print(tag, flush=True)
    conn.close()


if __name__ == "__main__":
    run()

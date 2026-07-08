import argparse
import sqlite3
import time

import pandas as pd

from src import nba_patch  # noqa: F401  patches nba_api before import
from nba_api.stats.endpoints import leaguedashlineups
from nba_api.stats.static import teams as static_teams

from src.config import DB_PATH, NBA_API_HEADERS, REQUEST_SLEEP, REQUEST_TIMEOUT, SEASONS

MAX_RETRIES = 7
BACKOFFS = [5, 15, 30, 60, 120, 240, 480]


def fetch_team_lineups(season: str, measure_type: str, team_id: int, group_quantity: int = 5) -> pd.DataFrame:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            df = leaguedashlineups.LeagueDashLineups(
                season=season,
                season_type_all_star="Regular Season",
                measure_type_detailed_defense=measure_type,
                per_mode_detailed="Totals",
                group_quantity=group_quantity,
                team_id_nullable=team_id,
                timeout=REQUEST_TIMEOUT,
                headers=NBA_API_HEADERS,
            ).get_data_frames()[0]
            df.insert(0, "SEASON", season)
            df.insert(1, "MEASURE_TYPE", measure_type)
            df.insert(2, "GROUP_QUANTITY", group_quantity)
            return df
        except Exception as e:
            last_err = e
            backoff = BACKOFFS[attempt]
            print(f"  retry {attempt+1}/{MAX_RETRIES} after {backoff}s ({type(e).__name__})", flush=True)
            time.sleep(backoff)
    raise last_err


def _ensure_group_quantity_column(conn: sqlite3.Connection, table: str) -> None:
    """Add GROUP_QUANTITY column to existing tables and backfill to 5 (the
    original hardcoded value). Safe to run repeatedly."""
    try:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    except sqlite3.OperationalError:
        return  # table doesn't exist yet — first run will create with column
    if "GROUP_QUANTITY" not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN GROUP_QUANTITY INTEGER")
        conn.execute(f"UPDATE {table} SET GROUP_QUANTITY=5 WHERE GROUP_QUANTITY IS NULL")
        conn.commit()
        print(f"  added GROUP_QUANTITY column to {table} (backfilled existing rows to 5)", flush=True)


def done_teams(conn: sqlite3.Connection, table: str, season: str, group_quantity: int) -> set:
    try:
        rows = conn.execute(
            f"SELECT DISTINCT TEAM_ID FROM {table} WHERE SEASON=? AND GROUP_QUANTITY=?",
            (season, group_quantity),
        ).fetchall()
        return {r[0] for r in rows}
    except sqlite3.OperationalError:
        return set()


def run(group_quantities=(2, 3, 5), measures=("Base",), seasons=None):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    teams = static_teams.get_teams()
    seasons = list(seasons) if seasons is not None else SEASONS
    conn = sqlite3.connect(DB_PATH)

    for measure in measures:
        table = f"lineup_stats_{measure.lower()}"
        _ensure_group_quantity_column(conn, table)

    for gq in group_quantities:
        for season in seasons:
            for measure in measures:
                table = f"lineup_stats_{measure.lower()}"
                already = done_teams(conn, table, season, gq)
                todo = [t for t in teams if t["id"] not in already]
                if not todo:
                    print(f"lineups gq={gq} {season} {measure}: skip (complete)", flush=True)
                    continue
                print(f"lineups gq={gq} {season} {measure}: {len(todo)} teams to fetch", flush=True)
                failed = []
                for team in todo:
                    try:
                        df = fetch_team_lineups(season, measure, team["id"], group_quantity=gq)
                    except Exception as e:
                        print(f"  SKIP {team['abbreviation']}: {type(e).__name__}", flush=True)
                        failed.append(team["abbreviation"])
                        continue
                    df.to_sql(table, conn, if_exists="append", index=False)
                    time.sleep(REQUEST_SLEEP)
                tag = f"  done gq={gq} {season} {measure}"
                if failed:
                    tag += f" (failed: {failed})"
                print(tag, flush=True)
    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape NBA lineup stats by group size.")
    parser.add_argument("--groups", type=int, nargs="+", default=[2, 3, 5],
                        help="Group quantities to fetch (default: 2 3 5).")
    parser.add_argument("--measures", nargs="+", default=["Base"],
                        choices=["Base", "Advanced"], help="Measure types.")
    parser.add_argument("--seasons", nargs="*", default=None,
                        help="Specific seasons (default: all from config).")
    args = parser.parse_args()
    run(group_quantities=args.groups, measures=args.measures, seasons=args.seasons)

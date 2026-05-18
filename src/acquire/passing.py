import json
import sqlite3
import sys
import time

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from src import nba_patch  # noqa: F401  patches nba_api before import
from nba_api.stats.endpoints import leaguedashplayerstats, playerdashptpass

from src.config import DB_PATH, REQUEST_SLEEP, REQUEST_TIMEOUT, SEASONS

PASSING_SEASONS = [s for s in SEASONS if s >= "2013-14"]
MIN_MINUTES = 100.0
MAX_RETRIES = 3
BACKOFFS = [5, 15, 30]
INTER_SEASON_SLEEP = 30
SKIP_COOLDOWN = 60          # sleep after a skip to let Akamai cool
HOT_STREAK_COOLDOWN = 300   # consecutive-skip threshold cooldown
HOT_STREAK_THRESHOLD = 3
SEASON_ABORT_STREAK = 6     # bail on season after this many consecutive skips

SESSION_BAD_ERRORS = (json.JSONDecodeError,)


def _retry(fn, *args, **kwargs):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt == 1 and isinstance(e, SESSION_BAD_ERRORS):
                nba_patch.reset_session()
            backoff = BACKOFFS[attempt]
            time.sleep(backoff)
    raise last_err


def fetch_player_index(season: str) -> pd.DataFrame:
    def _call():
        return leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star="Regular Season",
            per_mode_detailed="Totals",
            timeout=REQUEST_TIMEOUT,
        ).get_data_frames()[0]

    df = _retry(_call)
    return df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "MIN"]]


def fetch_passing(player_id: int, team_id: int, season: str):
    def _call():
        return playerdashptpass.PlayerDashPtPass(
            player_id=player_id,
            team_id=team_id,
            season=season,
            season_type_all_star="Regular Season",
            per_mode_simple="Totals",
            timeout=REQUEST_TIMEOUT,
        ).get_data_frames()

    return _retry(_call)


def done_players(conn: sqlite3.Connection, season: str) -> set:
    """Return (player_id, team_id) pairs already fetched for this season."""
    try:
        rows = conn.execute(
            "SELECT DISTINCT PLAYER_ID, TEAM_ID FROM passing_made WHERE SEASON=?", (season,)
        ).fetchall()
        return {(r[0], r[1]) for r in rows}
    except sqlite3.OperationalError:
        return set()


def run():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    for season in PASSING_SEASONS:
        players = fetch_player_index(season)
        players = players[players["MIN"] >= MIN_MINUTES].reset_index(drop=True)
        already = done_players(conn, season)
        todo = players[~players.apply(
            lambda r: (int(r["PLAYER_ID"]), int(r["TEAM_ID"])) in already, axis=1
        )].reset_index(drop=True) if already else players
        print(f"passing {season}: {len(todo)}/{len(players)} players to fetch", flush=True)
        time.sleep(REQUEST_SLEEP)

        skip_streak = 0
        ok_count = 0
        for _, p in todo.iterrows():
            try:
                made, received = fetch_passing(int(p["PLAYER_ID"]), int(p["TEAM_ID"]), season)
            except Exception as e:
                skip_streak += 1
                print(
                    f"  skip {season} {p['PLAYER_NAME']}: {type(e).__name__} "
                    f"(streak={skip_streak})", flush=True,
                )
                if skip_streak >= SEASON_ABORT_STREAK:
                    print(f"  ABORT {season}: {skip_streak} consecutive skips — moving on", flush=True)
                    break
                if skip_streak >= HOT_STREAK_THRESHOLD:
                    time.sleep(HOT_STREAK_COOLDOWN)
                else:
                    time.sleep(SKIP_COOLDOWN)
                nba_patch.reset_session()
                continue
            skip_streak = 0
            ok_count += 1
            for df, table in ((made, "passing_made"), (received, "passing_received")):
                if len(df):
                    df = df.copy()
                    df.insert(0, "SEASON", season)
                    df.to_sql(table, conn, if_exists="append", index=False)
            time.sleep(REQUEST_SLEEP)
        print(f"  done {season} (ok={ok_count})", flush=True)
        time.sleep(INTER_SEASON_SLEEP)
    conn.close()


if __name__ == "__main__":
    run()

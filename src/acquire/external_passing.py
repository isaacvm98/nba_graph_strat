"""Pull pre-computed passing-network CSVs from lucasholliday/nba-passing-networks.

Top-10 passers per team only; 2023-24 and 2024-25 seasons. Useful as a
fallback / prototyping dataset while our own nba_api pipeline is rate-limited.
"""
import io
import sqlite3
import time

import pandas as pd
import requests
from nba_api.stats.static import teams as static_teams

from src.config import DB_PATH

REPO_RAW = "https://raw.githubusercontent.com/lucasholliday/nba-passing-networks/main/data"
SEASONS = {"2023-24": "teams23", "2024-25": "teams24"}


def fetch_csv(url: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


def run():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS passing_external")
    conn.commit()

    teams = static_teams.get_teams()
    for season, folder in SEASONS.items():
        for team in teams:
            fname = team["full_name"].replace(" ", "_") + f"_{season}.csv"
            url = f"{REPO_RAW}/{folder}/{fname}"
            try:
                df = fetch_csv(url)
            except Exception as e:
                print(f"  miss {fname}: {type(e).__name__}", flush=True)
                continue
            df.insert(0, "SEASON", season)
            df.insert(1, "TEAM_FULL_NAME", team["full_name"])
            df.to_sql("passing_external", conn, if_exists="append", index=False)
            print(f"  {fname}: {len(df)} edges", flush=True)
            time.sleep(0.2)
    conn.close()


if __name__ == "__main__":
    run()

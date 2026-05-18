"""Extract network features per team-season into a flat table.

For each (season, team) we compute summary metrics on the passing and lineup
networks. Team-network metrics are season-level (not team-level) so they are
joined as season-wide features.

Output is a DataFrame that can be persisted via to_sql.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
from nba_api.stats.static import teams as static_teams

from src.config import DB_PATH, SEASONS
from src.network.builders import (
    build_lineup_network,
    build_passing_network,
    build_team_network,
)
from src.network.metrics import summarize
from src.network.utility_net import build_utility_network

PASSING_SEASONS = [s for s in SEASONS if s >= "2013-14"]
LINEUP_MIN_MINUTES = 10.0


def _prefix(d: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def compute_team_season_features(conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    own = conn is None
    conn = conn or sqlite3.connect(DB_PATH)
    try:
        teams = static_teams.get_teams()
        rows: list[dict] = []
        for season in SEASONS:
            team_g = build_team_network(season, conn=conn, k_neighbors=5)
            for team in teams:
                tid = team["id"]
                abbr = team["abbreviation"]
                row = {"season": season, "team_id": tid, "team_abbr": abbr}

                if season in PASSING_SEASONS:
                    pg = build_passing_network(season, abbr, conn=conn)
                    row.update(_prefix(summarize(pg), "pass"))
                else:
                    row.update(_prefix({k: None for k in [
                        "n_nodes","n_edges","n_components","efficiency",
                        "degree_entropy","n_communities","modularity_largest_share",
                        "resilience_drop3",
                    ]}, "pass"))

                lg = build_lineup_network(season, abbr, conn=conn, min_minutes=LINEUP_MIN_MINUTES)
                row.update(_prefix(summarize(lg), "lineup"))

                ug = build_utility_network(season, abbr, conn=conn)
                row.update(_prefix(summarize(ug), "util"))

                # Team-network features for this team's node:
                if tid in team_g:
                    deg = team_g.degree(tid, weight="weight")
                    row["team_node_degree"] = float(deg) if deg is not None else 0.0
                else:
                    row["team_node_degree"] = 0.0
                rows.append(row)
            print(f"features {season}: {len(teams)} teams done", flush=True)
        return pd.DataFrame(rows)
    finally:
        if own:
            conn.close()


def run():
    conn = sqlite3.connect(DB_PATH)
    df = compute_team_season_features(conn=conn)
    df.to_sql("team_season_features", conn, if_exists="replace", index=False)
    conn.close()
    print(f"wrote team_season_features: {len(df)} rows, {df.shape[1]} cols")


if __name__ == "__main__":
    run()

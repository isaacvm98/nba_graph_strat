"""Verification pass 3: prose claims in the report not covered by verify_claims.py.

  A. "Pattern replicates across teams (Jokic on DEN, Doncic on DAL, LeBron on LAL)"
     -> top-3 in_strength for DEN/DAL/LAL 2023-24.
  B. Mechanism: "contending teams play ~10 players regularly while tanking teams
     cycle 16+" -> n_nodes at min_mpg=12 for top-5 vs bottom-5 W_PCT teams,
     plus DET/MEM/BOS unfiltered node counts (notebook cites 30/33/19).
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DB_PATH


def passing_nodes_and_instrength(season, team, min_mpg=0.0):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT PLAYER_ID, PLAYER_NAME_LAST_FIRST, PASS_TEAMMATE_PLAYER_ID, "
        "PASS_TO, PASS FROM passing_made "
        "WHERE SEASON=? AND TEAM_ABBREVIATION=? AND PASS>0",
        conn, params=(season, team))
    if min_mpg > 0:
        mins = pd.read_sql_query(
            "SELECT PLAYER_ID, MIN FROM player_stats_base "
            "WHERE SEASON=? AND TEAM_ABBREVIATION=? AND MEASURE_TYPE='Base'",
            conn, params=(season, team))
        keep = set(mins.loc[mins["MIN"] >= min_mpg, "PLAYER_ID"].astype(int))
        df = df[df["PLAYER_ID"].astype(int).isin(keep)
                & df["PASS_TEAMMATE_PLAYER_ID"].astype(int).isin(keep)]
    conn.close()
    g = nx.DiGraph()
    names = {}
    for _, r in df.iterrows():
        names.setdefault(int(r["PLAYER_ID"]), r["PLAYER_NAME_LAST_FIRST"])
        names.setdefault(int(r["PASS_TEAMMATE_PLAYER_ID"]), r["PASS_TO"])
        g.add_edge(int(r["PLAYER_ID"]), int(r["PASS_TEAMMATE_PLAYER_ID"]),
                   weight=float(r["PASS"]))
    return g, names


print("A. Top-3 in_strength, 2023-24 (claim: Jokic DEN, Doncic DAL, LeBron LAL)")
for team in ["DEN", "DAL", "LAL"]:
    g, names = passing_nodes_and_instrength("2023-24", team)
    top = sorted(g.in_degree(weight="weight"), key=lambda t: -t[1])[:3]
    print(f"  {team}: " + " | ".join(f"{names[n]} ({int(s)})" for n, s in top))

print("\nB1. Unfiltered node counts 2023-24 (notebook cites DET=30, MEM=33, BOS=19)")
for team in ["DET", "MEM", "BOS"]:
    g, _ = passing_nodes_and_instrength("2023-24", team)
    print(f"  {team}: {g.number_of_nodes()} nodes")

print("\nB2. n_nodes at min_mpg=12: top-5 vs bottom-5 W_PCT, per season")
conn = sqlite3.connect(DB_PATH)
rows = []
for y in range(2013, 2026):
    season = f"{y}-{str(y+1)[-2:]}"
    wins = pd.read_sql_query(
        "SELECT TEAM_ID, W_PCT FROM team_stats_base "
        "WHERE SEASON=? AND MEASURE_TYPE='Base'", conn, params=(season,))
    teams = pd.read_sql_query(
        "SELECT DISTINCT TEAM_ID, TEAM_ABBREVIATION FROM passing_made WHERE SEASON=?",
        conn, params=(season,))
    m = teams.merge(wins, on="TEAM_ID").sort_values("W_PCT", ascending=False)
    nn = {}
    for _, r in m.iterrows():
        g, _ = passing_nodes_and_instrength(season, r["TEAM_ABBREVIATION"], min_mpg=12)
        nn[r["TEAM_ABBREVIATION"]] = g.number_of_nodes()
    abbrs = m["TEAM_ABBREVIATION"].tolist()
    top5 = np.mean([nn[a] for a in abbrs[:5]])
    bot5 = np.mean([nn[a] for a in abbrs[-5:]])
    rows.append({"season": season, "top5_nodes": top5, "bot5_nodes": bot5})
conn.close()
df = pd.DataFrame(rows)
print(df.round(1).to_string(index=False))
print(f"\n  mean top-5: {df['top5_nodes'].mean():.1f}   mean bottom-5: {df['bot5_nodes'].mean():.1f}")
print("DONE spot checks")

"""Verification pass 1: data integrity underlying the 2026-05-27 report claims.

Checks:
  A. Table row counts vs report §2.1
  B. Duplicate (season, team, passer, receiver) rows in passing_made
  C. Seasons/teams coverage in passing_made (13 seasons x 30 teams?)
  D. player_stats_base MIN semantics sanity (per-game -> max ~ 38-43, not 3000)
  E. Duplicate player rows (would break the min_mpg join)
  F. W_PCT completeness: W+L per team-season (2025-26 complete at 82?)
  G. lineup_stats_base GROUP_QUANTITY coverage per season
"""
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DB_PATH

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

conn = sqlite3.connect(DB_PATH)

print("=" * 70)
print("A. Row counts")
claims = {
    "passing_made": 94891,
    "passing_received": 94992,
    "passing_external": 5318,
    "player_stats_base": 7411,
    "team_stats_base": 420,
    "team_stats_advanced": 420,
}
tables = [r[0] for r in conn.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
for t in tables:
    n = conn.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0]
    claimed = claims.get(t)
    flag = ""
    if claimed is not None:
        flag = "  == claim OK" if n == claimed else f"  != CLAIMED {claimed}"
    print(f"  {t}: {n}{flag}")

print("=" * 70)
print("B. Duplicate passer->receiver rows in passing_made")
dups = pd.read_sql_query(
    """SELECT SEASON, TEAM_ABBREVIATION, PLAYER_ID, PASS_TEAMMATE_PLAYER_ID,
              COUNT(*) AS n, COUNT(DISTINCT PASS) AS n_distinct_pass
       FROM passing_made
       GROUP BY SEASON, TEAM_ABBREVIATION, PLAYER_ID, PASS_TEAMMATE_PLAYER_ID
       HAVING COUNT(*) > 1""", conn)
print(f"  duplicate keys: {len(dups)}")
if len(dups):
    print(f"  ... of which with CONFLICTING pass counts: {(dups['n_distinct_pass'] > 1).sum()}")
    print(dups.head(10).to_string(index=False))

print("=" * 70)
print("C. passing_made coverage: teams per season")
cov = pd.read_sql_query(
    """SELECT SEASON, COUNT(DISTINCT TEAM_ABBREVIATION) AS n_teams,
              COUNT(*) AS n_rows
       FROM passing_made GROUP BY SEASON ORDER BY SEASON""", conn)
print(cov.to_string(index=False))

print("=" * 70)
print("D. player_stats_base MIN semantics (MEASURE_TYPE='Base')")
mn = pd.read_sql_query(
    """SELECT MIN(MIN) AS min_min, MAX(MIN) AS max_min, AVG(MIN) AS avg_min
       FROM player_stats_base WHERE MEASURE_TYPE='Base'""", conn)
print(mn.to_string(index=False))
print("  (max ~ 38-43 => per-game; max ~ 3000 => totals. Claim requires per-game.)")

print("=" * 70)
print("E. Duplicate (SEASON, TEAM, PLAYER) in player_stats_base Base")
pdup = pd.read_sql_query(
    """SELECT SEASON, TEAM_ABBREVIATION, PLAYER_ID, COUNT(*) AS n
       FROM player_stats_base WHERE MEASURE_TYPE='Base'
       GROUP BY SEASON, TEAM_ABBREVIATION, PLAYER_ID HAVING COUNT(*) > 1""", conn)
print(f"  duplicates: {len(pdup)}")
if len(pdup):
    print(pdup.head(10).to_string(index=False))

print("=" * 70)
print("F. team_stats_base W+L per season (complete = 82 per team)")
wl = pd.read_sql_query(
    """SELECT SEASON, COUNT(*) AS n_teams, MIN(W+L) AS min_gp, MAX(W+L) AS max_gp
       FROM team_stats_base WHERE MEASURE_TYPE='Base'
       GROUP BY SEASON ORDER BY SEASON""", conn)
print(wl.to_string(index=False))

print("=" * 70)
print("G. lineup_stats_base coverage by GROUP_QUANTITY")
lq = pd.read_sql_query(
    """SELECT GROUP_QUANTITY, SEASON, COUNT(*) AS n_rows,
              COUNT(DISTINCT TEAM_ABBREVIATION) AS n_teams
       FROM lineup_stats_base
       GROUP BY GROUP_QUANTITY, SEASON ORDER BY GROUP_QUANTITY, SEASON""", conn)
print(lq.to_string(index=False))

conn.close()
print("\nDONE data integrity")

"""Construct the three networks per team-season.

1. Passing network — directed weighted graph of player-to-player passes.
2. Lineup network — undirected graph of 5-man lineups; edges between lineups
   that share >= SHARE_THRESHOLD players (weighted by # shared).
3. Team network — undirected graph of all 30 teams in a season; edges weighted
   by similarity (negative Euclidean distance) of advanced stat vectors.

The Guo et al. substitution network needs play-by-play; we use the shared-player
lineup network as a static proxy until/unless we fetch PBP.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.config import DB_PATH

SHARE_THRESHOLD = 3

# Numeric advanced features used for the team-similarity graph.
TEAM_SIM_FEATURES = [
    "OFF_RATING", "DEF_RATING", "NET_RATING",
    "AST_PCT", "AST_TO", "OREB_PCT", "DREB_PCT", "REB_PCT",
    "TM_TOV_PCT", "EFG_PCT", "TS_PCT", "PACE",
]


def _conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


def _parse_lineup_player_ids(group_id: str) -> tuple[int, ...]:
    parts = [p for p in group_id.split("-") if p]
    return tuple(sorted(int(p) for p in parts))


def build_passing_network(season: str, team_abbr: str, conn: sqlite3.Connection | None = None) -> nx.DiGraph:
    """Player-to-player passing graph for one team-season.

    Nodes carry PLAYER_NAME_LAST_FIRST as a 'name' attribute.
    Edges weighted by PASS count; also store AST as an edge attribute.
    """
    own_conn = conn is None
    conn = conn or _conn()
    try:
        df = pd.read_sql_query(
            "SELECT PLAYER_ID, PLAYER_NAME_LAST_FIRST, PASS_TEAMMATE_PLAYER_ID, "
            "PASS_TO, PASS, AST FROM passing_made "
            "WHERE SEASON=? AND TEAM_ABBREVIATION=?",
            conn,
            params=(season, team_abbr),
        )
    finally:
        if own_conn:
            conn.close()

    g = nx.DiGraph(season=season, team=team_abbr, layer="passing")
    if df.empty:
        return g

    names: dict[int, str] = {}
    for _, r in df.iterrows():
        names.setdefault(int(r["PLAYER_ID"]), r["PLAYER_NAME_LAST_FIRST"])
        # PASS_TO carries the teammate's name; teammate id is in PASS_TEAMMATE_PLAYER_ID
        names.setdefault(int(r["PASS_TEAMMATE_PLAYER_ID"]), r["PASS_TO"])
        u = int(r["PLAYER_ID"])
        v = int(r["PASS_TEAMMATE_PLAYER_ID"])
        passes = float(r["PASS"] or 0)
        ast = float(r["AST"] or 0)
        if passes <= 0:
            continue
        g.add_edge(u, v, weight=passes, ast=ast)

    for node in g.nodes:
        g.nodes[node]["name"] = names.get(node, str(node))
    return g


def build_lineup_network(
    season: str,
    team_abbr: str,
    conn: sqlite3.Connection | None = None,
    share_threshold: int = SHARE_THRESHOLD,
    min_minutes: float = 0.0,
) -> nx.Graph:
    """Shared-player network of 5-man lineups for one team-season.

    Nodes are 5-man lineups (keyed by GROUP_ID). Edge between two lineups iff
    they share at least `share_threshold` players; edge weight = #shared players.
    Each node carries playing time (MIN) and games played (GP).
    """
    own_conn = conn is None
    conn = conn or _conn()
    try:
        df = pd.read_sql_query(
            "SELECT GROUP_ID, GROUP_NAME, MIN, GP "
            "FROM lineup_stats_base "
            "WHERE SEASON=? AND TEAM_ABBREVIATION=? AND MIN>=?",
            conn,
            params=(season, team_abbr, min_minutes),
        )
    finally:
        if own_conn:
            conn.close()

    g = nx.Graph(season=season, team=team_abbr, layer="lineup")
    if df.empty:
        return g

    df = df.copy()
    df["players"] = df["GROUP_ID"].map(_parse_lineup_player_ids)
    for _, r in df.iterrows():
        g.add_node(
            r["GROUP_ID"],
            name=r["GROUP_NAME"],
            minutes=float(r["MIN"] or 0),
            gp=int(r["GP"] or 0),
            players=r["players"],
        )

    # Pairwise overlap
    rows = df.to_dict("records")
    for i in range(len(rows)):
        pi = set(rows[i]["players"])
        for j in range(i + 1, len(rows)):
            pj = rows[j]["players"]
            shared = pi.intersection(pj)
            if len(shared) >= share_threshold:
                g.add_edge(rows[i]["GROUP_ID"], rows[j]["GROUP_ID"], weight=len(shared))
    return g


def build_team_network(
    season: str,
    conn: sqlite3.Connection | None = None,
    features: list[str] | None = None,
    k_neighbors: int | None = 5,
) -> nx.Graph:
    """Cross-team similarity graph for a season (30 nodes).

    Nodes are teams; node features are the requested advanced columns.
    By default we make it sparse by connecting each team to its k nearest
    neighbours in feature space (negative Euclidean distance as weight).
    """
    features = features or TEAM_SIM_FEATURES
    own_conn = conn is None
    conn = conn or _conn()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM team_stats_advanced WHERE SEASON=?",
            conn,
            params=(season,),
        )
    finally:
        if own_conn:
            conn.close()

    g = nx.Graph(season=season, layer="team")
    if df.empty:
        return g

    feats = [f for f in features if f in df.columns]
    X = df[feats].astype(float).to_numpy()
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    team_ids = df["TEAM_ID"].astype(int).tolist()
    team_names = df["TEAM_NAME"].tolist()
    for tid, name, row in zip(team_ids, team_names, X):
        g.add_node(tid, name=name, **{f: float(v) for f, v in zip(feats, row)})

    # Pairwise distances on standardized vectors.
    diffs = Xz[:, None, :] - Xz[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dist, np.inf)

    if k_neighbors is None:
        # Complete graph weighted by similarity = 1 / (1 + dist).
        n = len(team_ids)
        for i in range(n):
            for j in range(i + 1, n):
                g.add_edge(team_ids[i], team_ids[j], weight=1.0 / (1.0 + dist[i, j]), distance=float(dist[i, j]))
    else:
        for i, src in enumerate(team_ids):
            nn = np.argsort(dist[i])[:k_neighbors]
            for j in nn:
                tgt = team_ids[j]
                if g.has_edge(src, tgt):
                    continue
                g.add_edge(src, tgt, weight=1.0 / (1.0 + dist[i, j]), distance=float(dist[i, j]))
    return g

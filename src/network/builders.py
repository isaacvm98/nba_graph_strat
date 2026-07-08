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
    share_threshold: int | None = None,
    min_minutes: float = 0.0,
    group_quantity: int = 5,
) -> nx.Graph:
    """Shared-player network of N-man lineups for one team-season.

    Nodes are N-man lineups (keyed by GROUP_ID, where N=`group_quantity`).
    Edge between two lineups iff they share at least `share_threshold` players;
    edge weight = #shared players. Each node carries playing time (MIN) and
    games played (GP).

    `share_threshold` defaults to `max(1, group_quantity - 2)` — i.e. 1 for
    2-man, 1 for 3-man, 3 for 5-man (the original SHARE_THRESHOLD). Caller
    can override.
    """
    if share_threshold is None:
        share_threshold = max(1, group_quantity - 2)
    own_conn = conn is None
    conn = conn or _conn()
    try:
        df = pd.read_sql_query(
            "SELECT GROUP_ID, GROUP_NAME, MIN, GP "
            "FROM lineup_stats_base "
            "WHERE SEASON=? AND TEAM_ABBREVIATION=? AND MIN>=? AND GROUP_QUANTITY=?",
            conn,
            params=(season, team_abbr, min_minutes, group_quantity),
        )
    finally:
        if own_conn:
            conn.close()

    g = nx.Graph(season=season, team=team_abbr, layer="lineup", group_quantity=group_quantity)
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


def build_player_cooccurrence_network(
    season: str,
    team_abbr: str,
    conn: sqlite3.Connection | None = None,
    min_minutes: float = 0.0,
    group_quantity: int = 2,
) -> nx.Graph:
    """Player-level co-occurrence graph derived from N-man lineups.

    Nodes are *players*; edge weight between two players = total minutes they
    appear together in any N-man unit (default N=2, so weight = joint minutes
    directly). For N=3 the same pair appears across 3-man combos — we sum.
    Cleaner than the lineup-level graph for player-chemistry analysis.
    """
    own_conn = conn is None
    conn = conn or _conn()
    try:
        df = pd.read_sql_query(
            "SELECT GROUP_ID, GROUP_NAME, MIN "
            "FROM lineup_stats_base "
            "WHERE SEASON=? AND TEAM_ABBREVIATION=? AND MIN>=? AND GROUP_QUANTITY=?",
            conn,
            params=(season, team_abbr, min_minutes, group_quantity),
        )
    finally:
        if own_conn:
            conn.close()

    g = nx.Graph(season=season, team=team_abbr, layer="player_cooccurrence",
                 group_quantity=group_quantity)
    if df.empty:
        return g

    from itertools import combinations

    # Map player_id -> last-seen name (parsed from GROUP_NAME if present).
    names: dict[int, str] = {}
    for _, r in df.iterrows():
        pids = _parse_lineup_player_ids(r["GROUP_ID"])
        # GROUP_NAME is " - "-joined player names in the same order as GROUP_ID.
        parts = [p.strip() for p in (r["GROUP_NAME"] or "").split(" - ")] if r["GROUP_NAME"] else []
        for pid, nm in zip(pids, parts):
            names.setdefault(pid, nm)
        minutes = float(r["MIN"] or 0)
        for a, b in combinations(sorted(pids), 2):
            if g.has_edge(a, b):
                g[a][b]["weight"] += minutes
            else:
                g.add_edge(a, b, weight=minutes)

    for nd in g.nodes:
        g.nodes[nd]["name"] = names.get(nd, str(nd))
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

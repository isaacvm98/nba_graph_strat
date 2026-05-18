"""Yu & Yang (2017) player utility network.

Per team-season:
  1. Take each player's normalized box-score feature vector.
  2. Utility score = mean of normalized features (within team).
  3. Degree sequence: d_i = ceil(N * util_i^lambda / sum util^lambda).
  4. Build adjacency via Havel-Hakimi (preserves degree sequence).
  5. Edge weights = Euclidean distance between connected players' feature vectors.
  6. Predict via network efficiency.

`lambda_exp` 2.35 follows the paper.
"""
from __future__ import annotations

import math
import sqlite3

import networkx as nx
import numpy as np
import pandas as pd

from src.config import DB_PATH

UTILITY_FEATURES = ["GP", "MIN", "PTS", "AST", "REB", "STL", "BLK"]
LAMBDA_EXP = 2.35


def _normalize(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(axis=0), x.max(axis=0)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return (x - lo) / rng


def _degrees_from_utility(util: np.ndarray, n: int, lambda_exp: float) -> list[int]:
    p = util ** lambda_exp
    denom = p.sum()
    if denom <= 0:
        return [0] * n
    return [int(math.ceil(n * pi / denom)) for pi in p]


def _havel_hakimi_edges(degrees: list[int]) -> list[tuple[int, int]]:
    """Standard Havel-Hakimi realization; returns edges as (i, j) by node index."""
    seq = [(d, i) for i, d in enumerate(degrees)]
    edges: list[tuple[int, int]] = []
    while True:
        seq = [s for s in seq if s[0] > 0]
        if not seq:
            return edges
        seq.sort(key=lambda x: -x[0])
        d0, n0 = seq[0]
        if d0 > len(seq) - 1:
            # Sequence not graphical; cap and continue.
            d0 = len(seq) - 1
            seq[0] = (d0, n0)
        for k in range(1, d0 + 1):
            dk, nk = seq[k]
            edges.append((n0, nk))
            seq[k] = (dk - 1, nk)
        seq[0] = (0, n0)


def build_utility_network(
    season: str,
    team_abbr: str,
    conn: sqlite3.Connection | None = None,
    features: list[str] | None = None,
    lambda_exp: float = LAMBDA_EXP,
    min_minutes: float = 5.0,
) -> nx.Graph:
    features = features or UTILITY_FEATURES
    own = conn is None
    conn = conn or sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(
            "SELECT PLAYER_ID, PLAYER_NAME, " + ", ".join(features) +
            " FROM player_stats_base WHERE SEASON=? AND TEAM_ABBREVIATION=? AND MIN>=?",
            conn,
            params=(season, team_abbr, min_minutes),
        )
    finally:
        if own:
            conn.close()

    g = nx.Graph(season=season, team=team_abbr, layer="utility")
    if df.empty:
        return g

    X = df[features].astype(float).to_numpy()
    Xn = _normalize(X)
    utility = Xn.mean(axis=1)

    pids = df["PLAYER_ID"].astype(int).tolist()
    names = df["PLAYER_NAME"].tolist()
    for pid, name, u, row in zip(pids, names, utility, X):
        g.add_node(pid, name=name, utility=float(u), **{f: float(v) for f, v in zip(features, row)})

    degrees = _degrees_from_utility(utility, n=len(pids), lambda_exp=lambda_exp)
    edges_idx = _havel_hakimi_edges(degrees)

    diffs = Xn[:, None, :] - Xn[None, :, :]
    dist = np.linalg.norm(diffs, axis=2)
    for i, j in edges_idx:
        d = float(dist[i, j])
        g.add_edge(pids[i], pids[j], weight=1.0 / (1.0 + d), distance=d)
    return g

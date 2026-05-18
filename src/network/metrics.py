"""Network metrics drawn from the seed papers.

- Network efficiency (Latora & Marchiori; Yu & Yang use it as a team strength
  proxy via inverse shortest-path lengths).
- Community structure via Clauset-Newman-Moore greedy modularity (Guo et al.
  use this to label lineups core / connector / peripheral).
- Degree entropy — how evenly distributed influence is across nodes.
- Resilience — drop in efficiency after removing the top-degree nodes (used to
  pick up the "robustness against player attack" idea from Dr. Yang's notes).
"""
from __future__ import annotations

import math
from typing import Iterable

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


def network_efficiency(g: nx.Graph, weight: str | None = "weight") -> float:
    """Global efficiency of a (possibly weighted) graph.

    For weighted graphs we treat ``weight`` as edge strength, so shortest-path
    cost is ``1/weight``. NetworkX's built-in `global_efficiency` ignores
    weights, so we roll our own.
    """
    if g.number_of_nodes() < 2:
        return 0.0
    if weight and any(weight in d for _, _, d in g.edges(data=True)):
        # Build a temp graph with cost = 1/weight.
        h = g.copy()
        for u, v, d in h.edges(data=True):
            w = d.get(weight, 1.0)
            d["_cost"] = 1.0 / w if w > 0 else math.inf
        lengths = dict(nx.all_pairs_dijkstra_path_length(h, weight="_cost"))
    else:
        lengths = dict(nx.all_pairs_shortest_path_length(g))
    n = g.number_of_nodes()
    total = 0.0
    for u in g.nodes:
        for v in g.nodes:
            if u == v:
                continue
            d = lengths.get(u, {}).get(v)
            if d is not None and d > 0:
                total += 1.0 / d
    return total / (n * (n - 1))


def degree_entropy(g: nx.Graph, weight: str | None = "weight") -> float:
    """Shannon entropy of the (weighted) degree distribution, nats."""
    degs = [max(d, 0.0) for _, d in g.degree(weight=weight)]
    total = sum(degs)
    if total <= 0:
        return 0.0
    probs = [d / total for d in degs if d > 0]
    return -sum(p * math.log(p) for p in probs)


def communities(g: nx.Graph, weight: str | None = "weight") -> list[set]:
    """Clauset-Newman-Moore communities; treats g as undirected if directed."""
    h = g.to_undirected() if g.is_directed() else g
    if h.number_of_nodes() == 0:
        return []
    return [set(c) for c in greedy_modularity_communities(h, weight=weight)]


def resilience(
    g: nx.Graph,
    k: int = 3,
    weight: str | None = "weight",
    attack: str = "degree",
) -> float:
    """Fractional drop in efficiency when the top-k nodes are removed.

    attack: 'degree' (highest weighted degree first) or 'random' (placeholder).
    Returns 1 - eff_after / eff_before; higher = less resilient.
    """
    if g.number_of_nodes() <= k:
        return 1.0
    base = network_efficiency(g, weight=weight)
    if base <= 0:
        return 0.0
    h = g.copy()
    if attack == "degree":
        ordered = sorted(h.degree(weight=weight), key=lambda nd: -nd[1])
        targets = [n for n, _ in ordered[:k]]
    else:
        raise ValueError(f"unknown attack: {attack}")
    h.remove_nodes_from(targets)
    after = network_efficiency(h, weight=weight)
    return 1.0 - after / base


def summarize(g: nx.Graph, weight: str | None = "weight") -> dict:
    """Compact feature dict for downstream modeling."""
    n_components = nx.number_weakly_connected_components(g) if g.is_directed() else nx.number_connected_components(g)
    comms = communities(g, weight=weight)
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "n_components": n_components,
        "efficiency": network_efficiency(g, weight=weight),
        "degree_entropy": degree_entropy(g, weight=weight),
        "n_communities": len(comms),
        "modularity_largest_share": (
            max((len(c) for c in comms), default=0) / g.number_of_nodes()
            if g.number_of_nodes() else 0.0
        ),
        "resilience_drop3": resilience(g, k=3, weight=weight),
    }


def all_pairs_summary(
    seasons: Iterable[str],
    team_abbrs: Iterable[str],
    builder,
) -> list[dict]:
    """Run a builder + summarize across a grid of (season, team)."""
    rows = []
    for season in seasons:
        for team in team_abbrs:
            g = builder(season, team)
            row = {"season": season, "team": team, **summarize(g)}
            rows.append(row)
    return rows

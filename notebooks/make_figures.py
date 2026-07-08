"""Render the three figures for the Yang email (meeting_logs/figs/).

  fig1: BOS 2023-24 passing network, community-aware layout (Yang's viz ask)
  fig2: per-feature mean Pearson r vs W_PCT across 13 seasons, with per-season
        spread — validated / new / contradicted features
  fig3: win-prob mapping + OOS model comparison bars
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from networkx.algorithms.community import louvain_communities

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DB_PATH

FIGS = ROOT / "meeting_logs" / "figs"
FIGS.mkdir(exist_ok=True)

# ---- fig1: community-aware layout ------------------------------------------
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(
    "SELECT PLAYER_ID, PLAYER_NAME_LAST_FIRST, PASS_TEAMMATE_PLAYER_ID, PASS_TO, PASS "
    "FROM passing_made WHERE SEASON='2023-24' AND TEAM_ABBREVIATION='BOS' AND PASS>0",
    conn)

g_raw, g_inv = nx.DiGraph(), nx.DiGraph()
names = {}
for _, r in df.iterrows():
    u, v = int(r["PLAYER_ID"]), int(r["PASS_TEAMMATE_PLAYER_ID"])
    names.setdefault(u, r["PLAYER_NAME_LAST_FIRST"])
    names.setdefault(v, r["PASS_TO"])
    p = float(r["PASS"])
    g_raw.add_edge(u, v, weight=p)
    g_inv.add_edge(u, v, weight=1.0 / p)

comms = louvain_communities(g_inv.to_undirected(), weight="weight", seed=0)
comm_of = {n: i for i, c in enumerate(comms) for n in c}

n_c = max(len(comms), 1)
centers = {i: np.array([np.cos(2 * np.pi * i / n_c), np.sin(2 * np.pi * i / n_c)]) * 2.5
           for i in range(n_c)}
pos = {}
for i, c in enumerate(comms):
    sub = g_inv.to_undirected().subgraph(c)
    if sub.number_of_nodes() == 1:
        (node,) = list(sub.nodes)
        pos[node] = centers[i]
        continue
    sub_pos = nx.spring_layout(sub, seed=7, k=0.5, weight="weight")
    for node, p in sub_pos.items():
        pos[node] = np.array(p) * 0.7 + centers[i]

in_strength = dict(g_raw.in_degree(weight="weight"))
node_sizes = [80 + in_strength.get(n, 0) / 6 for n in g_inv.nodes]
node_colors = [comm_of[n] for n in g_inv.nodes]
edge_widths = [0.3 + g_raw[u][v]["weight"] / 250 for u, v in g_inv.edges()]
labels = {n: names[n].split(",")[0].strip() for n in g_inv.nodes}

fig, ax = plt.subplots(figsize=(11, 8.5))
nx.draw_networkx_edges(g_inv, pos, width=edge_widths, alpha=0.25,
                       edge_color="#555", arrows=False, ax=ax)
nx.draw_networkx_nodes(g_inv, pos, node_size=node_sizes,
                       node_color=node_colors, cmap="tab10", alpha=0.9, ax=ax)
nx.draw_networkx_labels(g_inv, pos, labels=labels, font_size=9, ax=ax)
ax.set_title("BOS 2023-24 — Louvain communities on inverse-weighted (1/n_pass) passing graph\n"
             "node size = passes received; edge width = raw passes; color = community "
             f"(n={len(comms)})")
ax.axis("off")
plt.tight_layout()
plt.savefig(FIGS / "fig1_community_layout_BOS.png", dpi=160)
plt.close()
print("fig1 saved")

# ---- fig2: feature correlation summary ---------------------------------------
v2 = pd.read_sql_query("SELECT * FROM team_season_features_v2", conn)
conn.close()

FEATS = [
    ("avg_in_ast", "assist inflow / player", "new"),
    ("avg_in_strength", "pass inflow / player", "validated"),
    ("hhi_in", "inflow concentration (HHI)", "new"),
    ("top3_share_in", "top-3 inflow share", "new"),
    ("ast_per_pass", "assists per pass", "new"),
    ("betweenness_mean", "mean betweenness", "validated"),
    ("entropy_norm", "entropy (size-normalized)", "rescue"),
    ("n_comm_top9", "communities (fixed top-9)", "rescue"),
    ("n_communities", "n communities", "contradicted"),
    ("degree_entropy", "degree entropy", "contradicted"),
    ("n_nodes", "n rotation players", "contradicted"),
]
seasons = sorted(v2["season"].unique())
rows = []
for f, label, grp in FEATS:
    rs = [v2[v2["season"] == s][[f, "W_PCT"]].corr().iloc[0, 1] for s in seasons]
    rows.append({"feature": label, "group": grp,
                 "mean_r": np.mean(rs), "rs": rs})
COL = {"validated": "#1f77b4", "new": "#2ca02c",
       "rescue": "#7f7f7f", "contradicted": "#d62728"}

fig, ax = plt.subplots(figsize=(9, 6.5))
ypos = np.arange(len(rows))[::-1]
for y, r in zip(ypos, rows):
    ax.scatter(r["rs"], [y] * len(r["rs"]), color=COL[r["group"]], alpha=0.35, s=22)
    ax.scatter([r["mean_r"]], [y], color=COL[r["group"]], s=160, zorder=3,
               edgecolor="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels([r["feature"] for r in rows])
ax.set_xlabel("Pearson r vs W_PCT (small dots = 13 individual seasons; large = mean)")
ax.set_title("Passing-network features vs winning, 2013-14 → 2025-26 (min 12 MPG)")
handles = [plt.Line2D([], [], marker="o", linestyle="", color=c, label=l)
           for l, c in [("validated (May)", COL["validated"]), ("new (July)", COL["new"]),
                        ("size-normalized rescue", COL["rescue"]),
                        ("contradicted hypothesis", COL["contradicted"])]]
ax.legend(handles=handles, loc="lower right", fontsize=9)
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
plt.savefig(FIGS / "fig2_feature_correlations.png", dpi=160)
plt.close()
print("fig2 saved")

# ---- fig3: method + model comparison ------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
m_names = ["distance from\nbest team\n(benchmark idea)", "distance from\ntop-3 centroid",
           "rank by\npass inflow\nalone", "within-season\nlinear regression"]
m_vals = [0.199, 0.334, 0.531, 0.594]
bars = axes[0].bar(m_names, m_vals,
                   color=["#d62728", "#ff9896", "#aec7e8", "#1f77b4"])
axes[0].set_ylabel("Spearman ρ vs W_PCT (mean of 13 seasons)")
axes[0].set_title("Win-prob mapping methods (in-sample)")
axes[0].bar_label(bars, fmt="%.2f")
axes[0].grid(alpha=0.3, axis="y")

o_names = ["May model\n(2 features)", "July model\n(+ assists per pass)"]
o_vals = [0.567, 0.643]
bars = axes[1].bar(o_names, o_vals, color=["#aec7e8", "#2ca02c"], width=0.5)
axes[1].set_ylabel("Spearman ρ (expanding-window OOS, 12 test seasons)")
axes[1].set_title("Out-of-sample forward prediction")
axes[1].bar_label(bars, fmt="%.3f")
axes[1].set_ylim(0, 0.75)
axes[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(FIGS / "fig3_method_and_model.png", dpi=160)
plt.close()
print("fig3 saved")
print("DONE figures")

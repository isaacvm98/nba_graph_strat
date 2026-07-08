"""Model-strengthening pass (2026-07-07): new network features, 13-season harness.

Feature groups (all on passing network at min_mpg=12 unless noted):
  3. Concentration measured directly: gini_in, top3_share_in, hhi_in
  4. Assist-weighted edges: avg_in_ast, ast_per_pass, gini_in_ast
  5. Size-normalized rescues of the contradicted features:
       entropy_norm (= entropy / log n), and fixed top-9-roster variants
       (n_comm_top9, entropy_top9, avg_in_strength_top9)
  6. Implemented-but-never-validated: efficiency, resilience_drop3,
       util_efficiency, util_entropy (Yu & Yang utility net)

Outputs:
  A. Per-season Pearson r vs W_PCT (mean, std, n_pos/13) for every feature
  B. Redundancy: pooled within-season-z correlation of each new feature with
     avg_in_strength
  C. OOS expanding-window: baseline {avg_in_strength, betweenness_mean} vs
     baseline + each new feature (delta), plus best-subset combo
Writes the full per-team-season feature table to team_season_features_v2 in
graph.sqlite for reuse in the notebook.
"""
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DB_PATH
from src.network.metrics import degree_entropy, network_efficiency, resilience
from src.network.utility_net import build_utility_network

pd.set_option("display.width", 220)

ALL_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2013, 2026)]
MPG = 12
BASELINE = ["avg_in_strength", "betweenness_mean"]


def _load_passes(conn, season, team):
    return pd.read_sql_query(
        "SELECT PLAYER_ID, PASS_TEAMMATE_PLAYER_ID, PASS, AST FROM passing_made "
        "WHERE SEASON=? AND TEAM_ABBREVIATION=? AND PASS>0",
        conn, params=(season, team))


def _load_minutes(conn, season, team):
    return pd.read_sql_query(
        "SELECT PLAYER_ID, MIN, GP FROM player_stats_base "
        "WHERE SEASON=? AND TEAM_ABBREVIATION=? AND MEASURE_TYPE='Base'",
        conn, params=(season, team))


def _graphs(df, keep=None):
    """Build (raw, inverse, ast) DiGraphs from a passing_made frame."""
    if keep is not None:
        df = df[df["PLAYER_ID"].astype(int).isin(keep)
                & df["PASS_TEAMMATE_PLAYER_ID"].astype(int).isin(keep)]
    g_raw, g_inv, g_ast = nx.DiGraph(), nx.DiGraph(), nx.DiGraph()
    for _, r in df.iterrows():
        u, v = int(r["PLAYER_ID"]), int(r["PASS_TEAMMATE_PLAYER_ID"])
        p, a = float(r["PASS"]), float(r["AST"] or 0)
        g_raw.add_edge(u, v, weight=p)
        g_inv.add_edge(u, v, weight=1.0 / p)
        if a > 0:
            g_ast.add_edge(u, v, weight=a)
    return g_raw, g_inv, g_ast


def _louvain(g, seed=0):
    h = g.to_undirected() if g.is_directed() else g
    if h.number_of_nodes() == 0:
        return []
    return louvain_communities(h, weight="weight", seed=seed)


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


def team_features(conn, season, team):
    passes = _load_passes(conn, season, team)
    if passes.empty:
        return None
    mins = _load_minutes(conn, season, team)
    keep12 = set(mins.loc[mins["MIN"] >= MPG, "PLAYER_ID"].astype(int))

    g_raw, g_inv, g_ast = _graphs(passes, keep=keep12)
    if g_raw.number_of_nodes() < 4:
        return None

    in_s = np.array([d for _, d in g_raw.in_degree(weight="weight")])
    betw = nx.betweenness_centrality(g_inv, weight="weight")
    h_raw = g_raw.to_undirected()

    row = {
        "season": season, "team": team,
        # baseline (must reproduce prior harness)
        "avg_in_strength": float(in_s.mean()),
        "betweenness_mean": float(np.mean(list(betw.values()))),
        "n_nodes": g_raw.number_of_nodes(),
        "degree_entropy": degree_entropy(h_raw, weight="weight"),
        "n_communities": len(_louvain(g_inv)),
        # 3. concentration, measured directly
        "gini_in": gini(in_s),
        "top3_share_in": float(np.sort(in_s)[-3:].sum() / in_s.sum()) if in_s.sum() > 0 else 0.0,
        "hhi_in": float(((in_s / in_s.sum()) ** 2).sum()) if in_s.sum() > 0 else 0.0,
        # 5a. size-normalized entropy
        "entropy_norm": (degree_entropy(h_raw, weight="weight") / np.log(g_raw.number_of_nodes())
                         if g_raw.number_of_nodes() > 1 else 0.0),
        # 6. implemented-but-untested (raw passing graph)
        "efficiency": network_efficiency(h_raw, weight="weight"),
        "resilience_drop3": resilience(h_raw, k=3, weight="weight"),
    }

    # 4. assist-weighted
    if g_ast.number_of_edges() > 0:
        in_ast = np.array([d for _, d in g_ast.in_degree(weight="weight")])
        total_pass = sum(d["weight"] for _, _, d in g_raw.edges(data=True))
        total_ast = sum(d["weight"] for _, _, d in g_ast.edges(data=True))
        row["avg_in_ast"] = float(in_ast.mean())
        row["ast_per_pass"] = float(total_ast / total_pass) if total_pass > 0 else 0.0
        row["gini_in_ast"] = gini(in_ast)
    else:
        row["avg_in_ast"] = row["ast_per_pass"] = row["gini_in_ast"] = np.nan

    # 5b. fixed top-9 roster (by total minutes MIN*GP) — removes the churn confound
    mins = mins.copy()
    mins["TOT_MIN"] = mins["MIN"].astype(float) * mins["GP"].astype(float)
    top9 = set(mins.sort_values("TOT_MIN", ascending=False)
               .head(9)["PLAYER_ID"].astype(int))
    g9_raw, g9_inv, _ = _graphs(passes, keep=top9)
    if g9_raw.number_of_nodes() >= 4:
        in9 = np.array([d for _, d in g9_raw.in_degree(weight="weight")])
        row["n_comm_top9"] = len(_louvain(g9_inv))
        row["entropy_top9"] = degree_entropy(g9_raw.to_undirected(), weight="weight")
        row["avg_in_strength_top9"] = float(in9.mean())
    else:
        row["n_comm_top9"] = row["entropy_top9"] = row["avg_in_strength_top9"] = np.nan

    # 6b. utility network (Yu & Yang), efficiency + entropy
    ug = build_utility_network(season, team, conn=conn)
    if ug.number_of_nodes() >= 4:
        row["util_efficiency"] = network_efficiency(ug, weight="weight")
        row["util_entropy"] = degree_entropy(ug, weight="weight")
    else:
        row["util_efficiency"] = row["util_entropy"] = np.nan

    return row


# ---- build table --------------------------------------------------------------
conn = sqlite3.connect(DB_PATH)
t0 = time.time()
rows = []
for season in ALL_SEASONS:
    teams = pd.read_sql_query(
        "SELECT DISTINCT TEAM_ID, TEAM_ABBREVIATION FROM passing_made WHERE SEASON=?",
        conn, params=(season,))
    wins = pd.read_sql_query(
        "SELECT TEAM_ID, W_PCT FROM team_stats_base WHERE SEASON=? AND MEASURE_TYPE='Base'",
        conn, params=(season,))
    wmap = dict(zip(wins["TEAM_ID"].astype(int), wins["W_PCT"]))
    for _, r in teams.iterrows():
        row = team_features(conn, season, r["TEAM_ABBREVIATION"])
        if row is not None:
            row["W_PCT"] = wmap.get(int(r["TEAM_ID"]))
            rows.append(row)
    print(f"  {season} done ({time.time()-t0:.0f}s)", flush=True)

feat_df = pd.DataFrame(rows)
feat_df.to_sql("team_season_features_v2", conn, if_exists="replace", index=False)
conn.close()
print(f"\nwrote team_season_features_v2: {len(feat_df)} rows, {feat_df.shape[1]} cols")

NEW_FEATS = ["gini_in", "top3_share_in", "hhi_in",
             "avg_in_ast", "ast_per_pass", "gini_in_ast",
             "entropy_norm", "n_comm_top9", "entropy_top9", "avg_in_strength_top9",
             "efficiency", "resilience_drop3", "util_efficiency", "util_entropy"]
CONTEXT_FEATS = ["avg_in_strength", "betweenness_mean", "degree_entropy",
                 "n_communities", "n_nodes"]

# ---- A. per-season correlation --------------------------------------------------
print("\n" + "=" * 70)
print("A. Per-season Pearson r vs W_PCT (13 seasons, min_mpg=12)")
per_season = {}
for season in ALL_SEASONS:
    df = feat_df[feat_df["season"] == season]
    per_season[season] = {
        f: df[[f, "W_PCT"]].corr().iloc[0, 1]
        for f in CONTEXT_FEATS + NEW_FEATS if df[f].notna().sum() > 2
    }
ps = pd.DataFrame(per_season).T
summary = pd.DataFrame({
    "mean_r": ps.mean(), "std_r": ps.std(), "n_pos": (ps > 0).sum(),
}).loc[CONTEXT_FEATS + NEW_FEATS]
print(summary.round(3).to_string())

# ---- B. redundancy vs avg_in_strength --------------------------------------------
print("\n" + "=" * 70)
print("B. Redundancy: pooled within-season-z correlation with avg_in_strength")
z = feat_df.copy()
for f in CONTEXT_FEATS + NEW_FEATS + ["W_PCT"]:
    z[f] = z.groupby("season")[f].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
for f in NEW_FEATS:
    ok = z[[f, "avg_in_strength"]].dropna()
    print(f"  {f:22s} corr = {ok[f].corr(ok['avg_in_strength']):+.3f}")

# ---- C. OOS incremental test -----------------------------------------------------
print("\n" + "=" * 70)
print("C. OOS expanding-window: baseline vs baseline + one new feature")


def oos_rho(feats):
    rhos = []
    for i in range(1, len(ALL_SEASONS)):
        train = feat_df[feat_df["season"].isin(ALL_SEASONS[:i])].dropna(subset=feats)
        test = feat_df[feat_df["season"] == ALL_SEASONS[i]].dropna(subset=feats)
        if len(train) < 20 or len(test) < 10:
            continue
        Xtr = train[feats].to_numpy(); ytr = train["W_PCT"].to_numpy()
        Xte = test[feats].to_numpy(); yte = test["W_PCT"].to_numpy()
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0, ddof=0); sd[sd == 0] = 1.0
        pred = LinearRegression().fit((Xtr - mu) / sd, ytr).predict((Xte - mu) / sd)
        rhos.append(spearmanr(pred, yte).statistic)
    return np.array(rhos)


base = oos_rho(BASELINE)
print(f"  baseline {BASELINE}: mean={base.mean():.3f} std={base.std(ddof=1):.3f} "
      f"n_pos={(base>0).sum()}/{len(base)}")
results = []
for f in NEW_FEATS:
    r = oos_rho(BASELINE + [f])
    if len(r) == len(base):
        results.append({"feature": f, "mean_rho": r.mean(),
                        "delta": r.mean() - base.mean(),
                        "helps_n": int((r - base > 0).sum()), "n_tests": len(r)})
    else:
        results.append({"feature": f, "mean_rho": r.mean() if len(r) else np.nan,
                        "delta": np.nan, "helps_n": np.nan, "n_tests": len(r)})
res = pd.DataFrame(results).sort_values("delta", ascending=False)
print(res.round(4).to_string(index=False))

# best-subset: baseline + every feature with positive delta, jointly
pos_feats = [r["feature"] for r in results
             if isinstance(r["delta"], float) and not np.isnan(r["delta"]) and r["delta"] > 0]
if pos_feats:
    r = oos_rho(BASELINE + pos_feats)
    print(f"\n  baseline + all positive-delta feats {pos_feats}:")
    print(f"    mean={r.mean():.3f} std={r.std(ddof=1):.3f} n_pos={(r>0).sum()}/{len(r)}"
          f"  (baseline {base.mean():.3f})")

print("\nDONE feature expansion")

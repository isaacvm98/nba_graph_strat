"""Network-strengthening round 2 (2026-07-07): directed structure, hygiene,
partition quality, assist-graph centrality, robustness.

Groups:
  H (hygiene)   : per-game normalization of strength features (COVID game counts)
  D (directed)  : weighted reciprocity, mean in/out imbalance, flow hierarchy
  Q (quality)   : modularity of the Louvain partition (inverse-weighted graph)
  A (assist)    : betweenness on inverse-assist graph, PageRank HHI / max
  R (robustness): 3-feature model at min_mpg in {8, 12, 16}; RF vs linear OOS

Evaluation mirrors feature_expansion.py: per-season r (13 seasons), redundancy
vs the current 3-feature model, OOS incremental deltas.
Persists features to team_season_features_v3.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DB_PATH
from src.network.metrics import degree_entropy

pd.set_option("display.width", 220)

ALL_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2013, 2026)]
MODEL3 = ["avg_in_strength", "betweenness_mean", "ast_per_pass"]


def _load_passes(conn, season, team):
    return pd.read_sql_query(
        "SELECT PLAYER_ID, PASS_TEAMMATE_PLAYER_ID, PASS, AST FROM passing_made "
        "WHERE SEASON=? AND TEAM_ABBREVIATION=? AND PASS>0",
        conn, params=(season, team))


def _keep_set(conn, season, team, mpg):
    mins = pd.read_sql_query(
        "SELECT PLAYER_ID, MIN FROM player_stats_base "
        "WHERE SEASON=? AND TEAM_ABBREVIATION=? AND MEASURE_TYPE='Base'",
        conn, params=(season, team))
    return set(mins.loc[mins["MIN"] >= mpg, "PLAYER_ID"].astype(int))


def _graphs(df, keep):
    df = df[df["PLAYER_ID"].astype(int).isin(keep)
            & df["PASS_TEAMMATE_PLAYER_ID"].astype(int).isin(keep)]
    g_raw, g_inv, g_ast, g_ast_inv = nx.DiGraph(), nx.DiGraph(), nx.DiGraph(), nx.DiGraph()
    for _, r in df.iterrows():
        u, v = int(r["PLAYER_ID"]), int(r["PASS_TEAMMATE_PLAYER_ID"])
        p, a = float(r["PASS"]), float(r["AST"] or 0)
        g_raw.add_edge(u, v, weight=p)
        g_inv.add_edge(u, v, weight=1.0 / p)
        if a > 0:
            g_ast.add_edge(u, v, weight=a)
            g_ast_inv.add_edge(u, v, weight=1.0 / a)
    return g_raw, g_inv, g_ast, g_ast_inv


def _louvain(g, seed=0):
    h = g.to_undirected() if g.is_directed() else g
    if h.number_of_nodes() == 0:
        return []
    return louvain_communities(h, weight="weight", seed=seed)


def team_features(conn, season, team, gp, mpg=12):
    passes = _load_passes(conn, season, team)
    if passes.empty:
        return None
    keep = _keep_set(conn, season, team, mpg)
    g_raw, g_inv, g_ast, g_ast_inv = _graphs(passes, keep)
    if g_raw.number_of_nodes() < 4:
        return None

    in_s = np.array([d for _, d in g_raw.in_degree(weight="weight")])
    out_s = np.array([d for _, d in g_raw.out_degree(weight="weight")])
    betw = nx.betweenness_centrality(g_inv, weight="weight")
    total_pass = in_s.sum()
    total_ast = sum(d["weight"] for _, _, d in g_ast.edges(data=True))

    # weighted reciprocity: sum over unordered pairs of 2*min(w_uv, w_vu) / total
    recip_num = 0.0
    seen = set()
    for u, v, d in g_raw.edges(data=True):
        if (v, u) in seen or (u, v) in seen:
            continue
        seen.add((u, v))
        w_uv = d["weight"]
        w_vu = g_raw[v][u]["weight"] if g_raw.has_edge(v, u) else 0.0
        recip_num += 2.0 * min(w_uv, w_vu)
    reciprocity_w = recip_num / total_pass if total_pass > 0 else 0.0

    imb = np.abs(in_s - out_s) / np.maximum(in_s + out_s, 1e-9)
    comms = _louvain(g_inv)
    h_raw = g_raw.to_undirected()
    # nx modularity on the raw undirected graph, partition from inverse graph
    try:
        modularity = nx.community.modularity(h_raw, comms, weight="weight")
    except Exception:
        modularity = np.nan
    try:
        flow_h = nx.flow_hierarchy(g_raw)
    except Exception:
        flow_h = np.nan
    pr = nx.pagerank(g_raw, weight="weight", alpha=0.85)
    pr_v = np.array(list(pr.values()))

    row = {
        "season": season, "team": team,
        # current model
        "avg_in_strength": float(in_s.mean()),
        "betweenness_mean": float(np.mean(list(betw.values()))),
        "ast_per_pass": float(total_ast / total_pass) if total_pass > 0 else np.nan,
        # H hygiene
        "avg_in_strength_pg": float(in_s.mean() / gp) if gp else np.nan,
        "avg_in_ast_pg": float((total_ast / g_ast.number_of_nodes()) / gp)
                         if (gp and g_ast.number_of_nodes()) else np.nan,
        # D directed
        "reciprocity_w": float(reciprocity_w),
        "inout_imbalance": float(imb.mean()),
        "flow_hierarchy": float(flow_h) if flow_h == flow_h else np.nan,
        # Q quality
        "modularity": float(modularity) if modularity == modularity else np.nan,
        # A assist-graph centrality
        "ast_betw_mean": (float(np.mean(list(
            nx.betweenness_centrality(g_ast_inv, weight="weight").values())))
            if g_ast_inv.number_of_nodes() >= 4 else np.nan),
        "pr_hhi": float((pr_v ** 2).sum()),
        "pr_max": float(pr_v.max()),
    }
    return row


conn = sqlite3.connect(DB_PATH)
gp_df = pd.read_sql_query(
    "SELECT SEASON, TEAM_ID, W+L AS GP FROM team_stats_base WHERE MEASURE_TYPE='Base'", conn)
gp_map = {(r["SEASON"], int(r["TEAM_ID"])): int(r["GP"]) for _, r in gp_df.iterrows()}

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
        tid = int(r["TEAM_ID"])
        row = team_features(conn, season, r["TEAM_ABBREVIATION"],
                            gp=gp_map.get((season, tid)))
        if row is not None:
            row["W_PCT"] = wmap.get(tid)
            rows.append(row)
    print(f"  {season} done ({time.time()-t0:.0f}s)", flush=True)

feat_df = pd.DataFrame(rows)
feat_df.to_sql("team_season_features_v3", conn, if_exists="replace", index=False)

NEW = ["avg_in_strength_pg", "avg_in_ast_pg", "reciprocity_w", "inout_imbalance",
       "flow_hierarchy", "modularity", "ast_betw_mean", "pr_hhi", "pr_max"]

print("\n" + "=" * 70)
print("A. Per-season Pearson r vs W_PCT")
ps = {}
for season in ALL_SEASONS:
    df = feat_df[feat_df["season"] == season]
    ps[season] = {f: df[[f, "W_PCT"]].corr().iloc[0, 1]
                  for f in MODEL3 + NEW if df[f].notna().sum() > 2}
ps = pd.DataFrame(ps).T
summ = pd.DataFrame({"mean_r": ps.mean(), "std_r": ps.std(),
                     "n_pos": (ps > 0).sum()}).loc[MODEL3 + NEW]
print(summ.round(3).to_string())

print("\n" + "=" * 70)
print("B. Redundancy: max |within-season-z corr| vs the 3 model features")
z = feat_df.copy()
for f in MODEL3 + NEW + ["W_PCT"]:
    z[f] = z.groupby("season")[f].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0))
for f in NEW:
    cors = {m: abs(z[[f, m]].dropna()[f].corr(z[[f, m]].dropna()[m])) for m in MODEL3}
    worst = max(cors, key=cors.get)
    print(f"  {f:20s} max|corr| = {cors[worst]:.3f} (vs {worst})")


def oos_rho(feats, model="linear"):
    rhos = []
    for i in range(1, len(ALL_SEASONS)):
        train = feat_df[feat_df["season"].isin(ALL_SEASONS[:i])].dropna(subset=feats)
        test = feat_df[feat_df["season"] == ALL_SEASONS[i]].dropna(subset=feats)
        if len(train) < 20 or len(test) < 10:
            continue
        Xtr = train[feats].to_numpy(); ytr = train["W_PCT"].to_numpy()
        Xte = test[feats].to_numpy(); yte = test["W_PCT"].to_numpy()
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0, ddof=0); sd[sd == 0] = 1.0
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        if model == "linear":
            est = LinearRegression()
        else:
            est = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                        random_state=0, n_jobs=-1)
        pred = est.fit(Xtr, ytr).predict(Xte)
        rhos.append(spearmanr(pred, yte).statistic)
    return np.array(rhos)


print("\n" + "=" * 70)
print("C. OOS: 3-feature model + one new feature (linear)")
base = oos_rho(MODEL3)
print(f"  model3 baseline: mean={base.mean():.3f} std={base.std(ddof=1):.3f}")
res = []
for f in NEW:
    r = oos_rho(MODEL3 + [f])
    d = r.mean() - base.mean() if len(r) == len(base) else np.nan
    res.append({"feature": f, "mean_rho": r.mean() if len(r) else np.nan,
                "delta": d,
                "helps_n": int((r - base > 0).sum()) if len(r) == len(base) else np.nan})
print(pd.DataFrame(res).sort_values("delta", ascending=False).round(4).to_string(index=False))

print("\n" + "=" * 70)
print("D. Hygiene swap: per-game strength instead of season-total")
for feats, name in [
    (["avg_in_strength_pg", "betweenness_mean", "ast_per_pass"], "pg_swap"),
    (["avg_in_ast_pg", "betweenness_mean", "ast_per_pass"], "ast_pg_swap"),
]:
    r = oos_rho(feats)
    print(f"  {name:12s}: mean={r.mean():.3f} std={r.std(ddof=1):.3f} "
          f"(model3 {base.mean():.3f})")

print("\n" + "=" * 70)
print("E. Nonlinear check: RandomForest vs linear on model3 and on all features")
r_rf3 = oos_rho(MODEL3, model="rf")
allf = MODEL3 + [f for f in NEW if f not in ("flow_hierarchy",)]
r_lin_all = oos_rho(allf)
r_rf_all = oos_rho(allf, model="rf")
print(f"  linear model3   : {base.mean():.3f}")
print(f"  RF     model3   : {r_rf3.mean():.3f}")
print(f"  linear all-feats: {r_lin_all.mean():.3f}")
print(f"  RF     all-feats: {r_rf_all.mean():.3f}")

print("\n" + "=" * 70)
print("F. Robustness: model3 OOS at min_mpg in {8, 12, 16}")
for mpg in [8, 16]:
    rows_m = []
    t1 = time.time()
    for season in ALL_SEASONS:
        teams = pd.read_sql_query(
            "SELECT DISTINCT TEAM_ID, TEAM_ABBREVIATION FROM passing_made WHERE SEASON=?",
            conn, params=(season,))
        wins = pd.read_sql_query(
            "SELECT TEAM_ID, W_PCT FROM team_stats_base WHERE SEASON=? AND MEASURE_TYPE='Base'",
            conn, params=(season,))
        wmap = dict(zip(wins["TEAM_ID"].astype(int), wins["W_PCT"]))
        for _, r in teams.iterrows():
            tid = int(r["TEAM_ID"])
            row = team_features(conn, season, r["TEAM_ABBREVIATION"],
                                gp=gp_map.get((season, tid)), mpg=mpg)
            if row is not None:
                row["W_PCT"] = wmap.get(tid)
                rows_m.append(row)
    alt = pd.DataFrame(rows_m)
    saved = feat_df
    feat_df = alt
    r = oos_rho(MODEL3)
    feat_df = saved
    print(f"  min_mpg={mpg}: mean={r.mean():.3f} std={r.std(ddof=1):.3f} "
          f"({time.time()-t1:.0f}s)")
print(f"  min_mpg=12: mean={base.mean():.3f} (reference)")

conn.close()
print("\nDONE feature v3")

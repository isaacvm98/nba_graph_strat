"""Verification pass 2: recompute every number claimed in
meeting_logs/2026-05-27_passing_network_findings.md and diff against claims.

Sections:
  1. BOS 2023-24 weight-mode comparison (inverse -> 4 communities, raw -> 2)
  2. BOS 2023-24 node table (Tatum in_strength 3607, pagerank 0.142, ...)
  3. Per-season Pearson r vs W_PCT at min_mpg=12, 13 seasons  (report 5.2)
  4. min_mpg sweep means                                       (report 5.3)
  5. Win-prob mapping, 4 methods                               (report 5.4)
  6. OOS expanding window                                      (report 5.5)
  7. Lineup 2-man: per-feature corr + OOS pass/lineup/combined (report 5.6)
  8. Bonus: g2 vs g3 feature correlation on full-coverage seasons
"""
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import louvain_communities
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DB_PATH
from src.network.metrics import degree_entropy
from src.network.builders import build_player_cooccurrence_network

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

ALL_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2013, 2026)]
FEATURES = ["n_nodes", "n_communities", "avg_in_strength", "avg_degree",
            "degree_entropy", "betweenness_max", "betweenness_mean"]
VALIDATED_FEATS = ["avg_in_strength", "betweenness_mean"]


# ---- notebook-identical builders --------------------------------------------

def passing_graph(season, team, weight_mode="inverse", min_mpg=0.0):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT PLAYER_ID, PLAYER_NAME_LAST_FIRST, PASS_TEAMMATE_PLAYER_ID, "
        "PASS_TO, PASS FROM passing_made "
        "WHERE SEASON=? AND TEAM_ABBREVIATION=? AND PASS>0",
        conn, params=(season, team),
    )
    if min_mpg > 0:
        mins = pd.read_sql_query(
            "SELECT PLAYER_ID, MIN FROM player_stats_base "
            "WHERE SEASON=? AND TEAM_ABBREVIATION=? AND MEASURE_TYPE='Base'",
            conn, params=(season, team),
        )
        keep = set(mins.loc[mins["MIN"] >= min_mpg, "PLAYER_ID"].astype(int))
        df = df[
            df["PLAYER_ID"].astype(int).isin(keep)
            & df["PASS_TEAMMATE_PLAYER_ID"].astype(int).isin(keep)
        ]
    conn.close()

    g = nx.DiGraph(season=season, team=team)
    if df.empty:
        return g
    n = df["PASS"].astype(float).to_numpy()
    if weight_mode == "raw":
        w = n
    elif weight_mode == "log":
        w = np.log1p(n)
    elif weight_mode == "zscore":
        mu, sd = n.mean(), n.std(ddof=0) or 1.0
        w = (n - mu) / sd
        w = w - w.min() + 1e-6
    elif weight_mode == "inverse":
        w = 1.0 / n
    else:
        raise ValueError(weight_mode)
    names = {}
    for (pid, pname, tid, tname), wi, ni in zip(
        df[["PLAYER_ID", "PLAYER_NAME_LAST_FIRST",
            "PASS_TEAMMATE_PLAYER_ID", "PASS_TO"]].itertuples(index=False, name=None),
        w, n,
    ):
        names.setdefault(int(pid), pname)
        names.setdefault(int(tid), tname)
        g.add_edge(int(pid), int(tid), weight=float(wi), n_pass=float(ni))
    for nd in g.nodes:
        g.nodes[nd]["name"] = names.get(nd, str(nd))
    return g


def louvain_partition(g, weight="weight", seed=0):
    h = g.to_undirected() if g.is_directed() else g
    if h.number_of_nodes() == 0:
        return []
    return louvain_communities(h, weight=weight, seed=seed)


def team_passing_features(season, team, min_mpg=0.0):
    g_raw = passing_graph(season, team, weight_mode="raw", min_mpg=min_mpg)
    g_inv = passing_graph(season, team, weight_mode="inverse", min_mpg=min_mpg)
    if g_raw.number_of_nodes() == 0:
        return None
    comms = louvain_partition(g_inv, seed=0)
    h_raw = g_raw.to_undirected()
    in_s = [d for _, d in g_raw.in_degree(weight="weight")]
    out_s = [d for _, d in g_raw.out_degree(weight="weight")]
    degs = [d for _, d in g_raw.degree()]
    betw = nx.betweenness_centrality(g_inv, weight="weight")
    return {
        "season": season, "team": team, "min_mpg": min_mpg,
        "n_nodes": g_raw.number_of_nodes(),
        "n_edges": g_raw.number_of_edges(),
        "n_communities": len(comms),
        "avg_in_strength": float(np.mean(in_s)),
        "avg_out_strength": float(np.mean(out_s)),
        "avg_degree": float(np.mean(degs)),
        "degree_entropy": degree_entropy(h_raw, weight="weight"),
        "betweenness_max": float(max(betw.values())) if betw else 0.0,
        "betweenness_mean": float(np.mean(list(betw.values()))) if betw else 0.0,
    }


def season_team_features(season, min_mpg=0.0):
    conn = sqlite3.connect(DB_PATH)
    teams_in_season = pd.read_sql_query(
        "SELECT DISTINCT TEAM_ID, TEAM_ABBREVIATION FROM passing_made WHERE SEASON=?",
        conn, params=(season,),
    )
    team_wins = pd.read_sql_query(
        "SELECT TEAM_ID, TEAM_NAME, W, L, W_PCT FROM team_stats_base "
        "WHERE SEASON=? AND MEASURE_TYPE='Base'",
        conn, params=(season,),
    )
    conn.close()
    rows = []
    for _, r in teams_in_season.iterrows():
        feats = team_passing_features(season, r["TEAM_ABBREVIATION"], min_mpg=min_mpg)
        if feats is not None:
            feats["TEAM_ID"] = int(r["TEAM_ID"])
            rows.append(feats)
    return (
        pd.DataFrame(rows)
        .merge(team_wins, on="TEAM_ID", how="left")
        .sort_values("W_PCT", ascending=False)
        .reset_index(drop=True)
    )


# ---- 1. weight-mode comparison ----------------------------------------------
print("=" * 70)
print("1. BOS 2023-24 weight-mode comparison (claim: inverse=4 comms, raw=2)")
for mode in ["raw", "log", "zscore", "inverse"]:
    g = passing_graph("2023-24", "BOS", weight_mode=mode)
    comms = louvain_partition(g, seed=0)
    print(f"  {mode:8s}: n_nodes={g.number_of_nodes()}, "
          f"n_communities={len(comms)}, "
          f"largest_share={max((len(c) for c in comms), default=0)/max(g.number_of_nodes(),1):.2f}")

# ---- 2. BOS node table --------------------------------------------------------
print("=" * 70)
print("2. BOS 2023-24 node table (claim: Tatum 3607/0.142/comm2, White 3361/0.135, ...)")
g_raw = passing_graph("2023-24", "BOS", weight_mode="raw")
g_inv = passing_graph("2023-24", "BOS", weight_mode="inverse")
comms = louvain_partition(g_inv, seed=0)
comm_of = {n: i for i, c in enumerate(comms) for n in c}
pr = nx.pagerank(g_raw, weight="weight", alpha=0.85)
node_df = pd.DataFrame([
    {"player": g_raw.nodes[n]["name"],
     "community": comm_of.get(n, -1),
     "in_strength": g_raw.in_degree(n, weight="weight"),
     "pagerank": pr.get(n, 0.0)}
    for n in g_raw.nodes
]).sort_values("in_strength", ascending=False).head(6)
print(node_df.round(3).to_string(index=False))

# ---- 3+4. cross-season features ----------------------------------------------
print("=" * 70)
print("3+4. Building features for 13 seasons x 5 thresholds (slow)...")
t0 = time.time()
season_features = {}
for season in ALL_SEASONS:
    for mpg in [0, 8, 12, 16, 20]:
        season_features[(season, mpg)] = season_team_features(season, min_mpg=mpg)
    print(f"  {season} done ({time.time()-t0:.0f}s)", flush=True)

print("-" * 70)
print("3. Per-season r at min_mpg=12 -> stability summary (report 5.2)")
per_season = []
for season in ALL_SEASONS:
    df = season_features[(season, 12)]
    row = {"season": season}
    for feat in FEATURES:
        if df[feat].notna().sum() > 2:
            row[feat] = df[[feat, "W_PCT"]].corr().iloc[0, 1]
    per_season.append(row)
per_season_df = pd.DataFrame(per_season).set_index("season")
print(per_season_df.round(2).to_string())
summary = pd.DataFrame({
    "mean_r": per_season_df.mean(),
    "std_r": per_season_df.std(),
    "n_pos": (per_season_df > 0).sum(),
})
print("\nStability summary (compare to report 5.2 table):")
print(summary.round(3).to_string())
CLAIM_52 = {
    "avg_in_strength": (0.55, 13), "betweenness_mean": (0.32, 13),
    "betweenness_max": (0.11, 11), "n_communities": (-0.32, 1),
    "n_nodes": (-0.52, 0), "degree_entropy": (-0.50, 0), "avg_degree": (-0.45, 0),
}
print("\nDiff vs claimed:")
for f, (r_claim, npos_claim) in CLAIM_52.items():
    r_got = summary.loc[f, "mean_r"]
    npos_got = int(summary.loc[f, "n_pos"])
    ok = (abs(r_got - r_claim) < 0.005) and (npos_got == npos_claim)
    print(f"  {f:20s} claimed r={r_claim:+.2f} ({npos_claim}/13)  "
          f"got r={r_got:+.3f} ({npos_got}/13)  {'OK' if ok else '<-- MISMATCH'}")

print("-" * 70)
print("4. min_mpg sweep, mean r across seasons (report 5.3)")
cells = []
for season in ALL_SEASONS:
    for mpg in [0, 8, 12, 16, 20]:
        df = season_features[(season, mpg)]
        row = {"season": season, "min_mpg": mpg}
        for feat in FEATURES:
            if df[feat].notna().sum() > 2:
                row[feat] = df[[feat, "W_PCT"]].corr().iloc[0, 1]
        cells.append(row)
cells_df = pd.DataFrame(cells)
mean_r = cells_df.groupby("min_mpg")[FEATURES].mean()
print(mean_r[["avg_in_strength", "betweenness_mean", "degree_entropy"]].round(3).to_string())
print("Claimed: mpg=0: .38/.34/-.55 | 8: .50/.32/-.44 | 12: .55/.32/-.50 | 16: .53/.23/-.40 | 20: .52/.08/-.26")

# ---- 5. win-prob mapping ------------------------------------------------------
print("=" * 70)
print("5. Win-prob mapping, 4 methods (report 5.4)")

def score_season(df, method, feats=VALIDATED_FEATS):
    df = df.copy()
    X = df[feats].to_numpy(dtype=float)
    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=0); sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    y = df["W_PCT"].to_numpy()
    if method == "single-feature":
        return Z[:, feats.index("avg_in_strength")]
    if method == "dist-from-best-1":
        anchor = Z[int(np.argmax(y))]
        return -np.linalg.norm(Z - anchor, axis=1)
    if method == "dist-from-top3":
        top3 = np.argsort(-y)[:3]
        anchor = Z[top3].mean(axis=0)
        return -np.linalg.norm(Z - anchor, axis=1)
    if method == "linreg":
        reg = LinearRegression().fit(Z, y)
        return reg.predict(Z)
    raise ValueError(method)

methods = ["single-feature", "dist-from-best-1", "dist-from-top3", "linreg"]
per_season_rhos = {m: [] for m in methods}
for season in ALL_SEASONS:
    df = season_features[(season, 12)]
    for m in methods:
        rho = spearmanr(score_season(df, m), df["W_PCT"]).statistic
        per_season_rhos[m].append(rho)
cmp5 = pd.DataFrame({
    m: {"mean_rho": np.mean(v), "std_rho": np.std(v, ddof=1),
        "min_rho": np.min(v), "n_pos": int(np.sum(np.array(v) > 0))}
    for m, v in per_season_rhos.items()
}).T
print(cmp5.round(3).to_string())
print("Claimed: linreg .594/13 | single .531/13 | top3 .334/12 | best-1 .199/9 (min -0.47)")

# ---- 6. OOS expanding window --------------------------------------------------
print("=" * 70)
print("6. OOS expanding window (report 5.5)")

def _within_season_z(X):
    mu = X.mean(axis=0); sd = X.std(axis=0, ddof=0); sd[sd == 0] = 1.0
    return (X - mu) / sd

def _standardize_train_test(X_train, X_test):
    mu = X_train.mean(axis=0); sd = X_train.std(axis=0, ddof=0); sd[sd == 0] = 1.0
    return (X_train - mu) / sd, (X_test - mu) / sd

oos = []
for i, test_s in enumerate(ALL_SEASONS):
    test_df = season_features[(test_s, 12)]
    y_te = test_df["W_PCT"].to_numpy()
    X_in = _within_season_z(test_df[VALIDATED_FEATS].to_numpy())
    rho_in = spearmanr(LinearRegression().fit(X_in, y_te).predict(X_in), y_te).statistic
    row = {"season": test_s, "rho_in_sample": rho_in,
           "rho_single_feat": spearmanr(test_df["avg_in_strength"], y_te).statistic}
    if i > 0:
        train_seasons = ALL_SEASONS[:i]
        train_df = pd.concat([season_features[(s, 12)] for s in train_seasons], ignore_index=True)
        y_tr = train_df["W_PCT"].to_numpy()
        X_tr, X_te = _standardize_train_test(
            train_df[VALIDATED_FEATS].to_numpy(), test_df[VALIDATED_FEATS].to_numpy())
        row["rho_oos_pooled"] = spearmanr(
            LinearRegression().fit(X_tr, y_tr).predict(X_te), y_te).statistic
        X_tr_w = np.vstack([
            _within_season_z(season_features[(s, 12)][VALIDATED_FEATS].to_numpy())
            for s in train_seasons])
        X_te_w = _within_season_z(test_df[VALIDATED_FEATS].to_numpy())
        row["rho_oos_within"] = spearmanr(
            LinearRegression().fit(X_tr_w, y_tr).predict(X_te_w), y_te).statistic
    oos.append(row)
oos_df = pd.DataFrame(oos).set_index("season")
mcols = ["rho_in_sample", "rho_oos_pooled", "rho_oos_within", "rho_single_feat"]
agg = oos_df.iloc[1:][mcols].agg(["mean", "std", "min"]).round(3)
agg.loc["n_pos"] = (oos_df.iloc[1:][mcols] > 0).sum()
print(agg.to_string())
print("Claimed: in-sample .600/12 | OOS pooled .567/12 | OOS within .569/12 | single .531/12")

# ---- 7. lineup ---------------------------------------------------------------
print("=" * 70)
print("7. Lineup 2-man co-occurrence (report 5.6)")

LINUP_FEATS = ["linup_avg_strength", "linup_max_strength", "linup_entropy",
               "linup_n_comm", "linup_n_nodes", "linup_density"]

def lineup_team_features(season, team, group_quantity=2, min_minutes=5.0):
    g = build_player_cooccurrence_network(season, team, min_minutes=min_minutes,
                                          group_quantity=group_quantity)
    if g.number_of_nodes() == 0:
        return None
    strengths = [d for _, d in g.degree(weight="weight")]
    comms = louvain_partition(g, seed=0)
    return {
        "linup_avg_strength": float(np.mean(strengths)),
        "linup_max_strength": float(np.max(strengths)),
        "linup_entropy": degree_entropy(g, weight="weight"),
        "linup_n_comm": len(comms),
        "linup_n_nodes": g.number_of_nodes(),
        "linup_density": nx.density(g),
    }

lineup_pool = {}
t0 = time.time()
for s in ALL_SEASONS:
    pass_df = season_features[(s, 12)]
    rows = []
    for _, r in pass_df.iterrows():
        lf = lineup_team_features(s, r["team"], group_quantity=2)
        if lf is None:
            continue
        rows.append({**r.to_dict(), **lf})
    lineup_pool[s] = pd.DataFrame(rows)
print(f"  built in {time.time()-t0:.0f}s; teams/season: {sorted(set(len(v) for v in lineup_pool.values()))}")

rows = []
for s in ALL_SEASONS:
    df = lineup_pool[s]
    row = {"season": s}
    for f in VALIDATED_FEATS + LINUP_FEATS:
        if df[f].notna().sum() > 2:
            row[f] = df[[f, "W_PCT"]].corr().iloc[0, 1]
    rows.append(row)
corr_df = pd.DataFrame(rows).set_index("season")
summary7 = pd.DataFrame({
    "mean_r": corr_df.mean(), "std_r": corr_df.std(),
    "n_pos": (corr_df > 0).sum()})
print(summary7.round(3).to_string())
print("Claimed: linup_max_strength +.40 (13) | avg_strength +.36 (13) | density +.16 (11)")
print("         linup_entropy -.56 (0) | n_nodes -.36 (0) | n_comm -.22 (1)")

def oos_predict(feats_to_use):
    rhos = []
    for i in range(1, len(ALL_SEASONS)):
        train = pd.concat([lineup_pool[s] for s in ALL_SEASONS[:i]], ignore_index=True)
        test = lineup_pool[ALL_SEASONS[i]]
        Xtr = train[feats_to_use].to_numpy(); ytr = train["W_PCT"].to_numpy()
        Xte = test[feats_to_use].to_numpy(); yte = test["W_PCT"].to_numpy()
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0, ddof=0); sd[sd == 0] = 1.0
        Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
        pred = LinearRegression().fit(Xtr, ytr).predict(Xte)
        rhos.append(spearmanr(pred, yte).statistic)
    return np.array(rhos)

r_pass = oos_predict(VALIDATED_FEATS)
r_lin = oos_predict(LINUP_FEATS)
r_comb = oos_predict(VALIDATED_FEATS + LINUP_FEATS)
for name, r in [("pass_only", r_pass), ("linup_only", r_lin), ("combined", r_comb)]:
    print(f"  {name:10s}: mean={r.mean():.3f} std={r.std(ddof=1):.3f} "
          f"min={r.min():.2f} max={r.max():.2f} n_pos={(r>0).sum()}/12")
delta = r_comb - r_pass
print(f"  mean delta (combined - pass_only) = {delta.mean():+.4f}  "
      f"(claimed +0.0024); lineup helps in {(delta>0).sum()}/12 (claimed 7/12)")

# ---- 8. bonus: g2 vs g3 ------------------------------------------------------
print("=" * 70)
print("8. Bonus: g2 vs g3 feature correlation (claimed ~1.0 on overlap seasons)")
G3_SEASONS = [s for s in ALL_SEASONS if s <= "2021-22"]
rows2, rows3 = [], []
for s in G3_SEASONS:
    for _, r in season_features[(s, 12)].iterrows():
        f2 = lineup_team_features(s, r["team"], group_quantity=2)
        f3 = lineup_team_features(s, r["team"], group_quantity=3)
        if f2 and f3:
            rows2.append({"season": s, "team": r["team"], **f2})
            rows3.append({"season": s, "team": r["team"], **f3})
df2 = pd.DataFrame(rows2); df3 = pd.DataFrame(rows3)
for f in LINUP_FEATS:
    c = np.corrcoef(df2[f], df3[f])[0, 1]
    print(f"  {f:22s} corr(g2, g3) = {c:.3f}   (n={len(df2)} team-seasons)")

print("\nDONE claims verification")

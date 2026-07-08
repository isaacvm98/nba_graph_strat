"""Follow-up: honest OOS comparison of a-priori feature combinations.

Reads team_season_features_v2 (written by feature_expansion.py). Combos chosen
on prior grounds (strength + low redundancy), not by ranking OOS deltas:
assist features measure pass *productivity* (ast_per_pass corr with
avg_in_strength is only +0.05).
"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import DB_PATH

ALL_SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2013, 2026)]

conn = sqlite3.connect(DB_PATH)
feat_df = pd.read_sql_query("SELECT * FROM team_season_features_v2", conn)
conn.close()


def oos_detail(feats):
    rhos = {}
    for i in range(1, len(ALL_SEASONS)):
        train = feat_df[feat_df["season"].isin(ALL_SEASONS[:i])].dropna(subset=feats)
        test = feat_df[feat_df["season"] == ALL_SEASONS[i]].dropna(subset=feats)
        Xtr = train[feats].to_numpy(); ytr = train["W_PCT"].to_numpy()
        Xte = test[feats].to_numpy(); yte = test["W_PCT"].to_numpy()
        mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0, ddof=0); sd[sd == 0] = 1.0
        pred = LinearRegression().fit((Xtr - mu) / sd, ytr).predict((Xte - mu) / sd)
        rhos[ALL_SEASONS[i]] = spearmanr(pred, yte).statistic
    return pd.Series(rhos)


COMBOS = {
    "A_baseline(pass_vol)": ["avg_in_strength", "betweenness_mean"],
    "B_base+ast_per_pass": ["avg_in_strength", "betweenness_mean", "ast_per_pass"],
    "C_base+both_ast": ["avg_in_strength", "betweenness_mean", "avg_in_ast", "ast_per_pass"],
    "D_ast_replaces_vol": ["avg_in_ast", "ast_per_pass", "betweenness_mean"],
    "E_ast_only": ["avg_in_ast", "ast_per_pass"],
    "F_D+hhi_in": ["avg_in_ast", "ast_per_pass", "betweenness_mean", "hhi_in"],
}

detail = pd.DataFrame({name: oos_detail(feats) for name, feats in COMBOS.items()})
print("Per-season OOS Spearman rho:")
print(detail.round(3).to_string())
print("\nAggregates (12 test seasons):")
agg = detail.agg(["mean", "std", "min"]).round(3)
agg.loc["n_pos"] = (detail > 0).sum()
agg.loc["beats_A_n"] = [(detail[c] > detail["A_baseline(pass_vol)"]).sum() for c in detail.columns]
print(agg.to_string())
print("\nDONE combos")

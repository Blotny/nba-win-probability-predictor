"""Score every game in games_final.csv with the saved model and write flat tables
for the Power BI dashboard:

  data/processed/predictions.csv       one row per game: probs, outcome, error terms
  data/processed/team_elo_history.csv  long: team, date, pre-game Elo
  data/processed/calibration.csv       split x probability-bucket: n, mean_pred, mean_actual

Only the 2025-26 season (split == "test") is genuine out-of-sample; the model was
fit on 2020-21 .. 2024-25.

Run from anywhere:  python src/predict.py   (needs models/win_prob_blend.joblib)
"""
import os
import sys

import numpy as np
import pandas as pd
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.train_model import load_and_split_data  # noqa: E402

DATA_PATH = os.path.join(ROOT, "data", "processed", "games_final.csv")
MODEL_PATH = os.path.join(ROOT, "models", "win_prob_blend.joblib")
OUT_DIR = os.path.join(ROOT, "data", "processed")

TEST_SEASONS = [22025]
VAL_SEASONS = [22024]
SEASON_LABEL = {22020: "2020-21", 22021: "2021-22", 22022: "2022-23",
                22023: "2023-24", 22024: "2024-25", 22025: "2025-26"}
EPS = 1e-6


def _split(sid):
    if sid in TEST_SEASONS:
        return "test"
    if sid in VAL_SEASONS:
        return "val"
    return "train"


def main():
    bundle = joblib.load(MODEL_PATH)
    f = bundle["features"]
    X, y, y_margin, meta = load_and_split_data(DATA_PATH)

    p_lr = bundle["logreg"].predict_proba(
        bundle["scaler"].transform(X[f].fillna(bundle["medians"])))[:, 1]
    p_xgb = bundle["xgb"].predict_proba(X[f])[:, 1]
    p = bundle["blend_w"] * p_lr + (1 - bundle["blend_w"]) * p_xgb

    pred_margin = bundle["xgb_margin"].predict(X[f])
    margin_win_prob = bundle["margin_link"].predict_proba(pred_margin.reshape(-1, 1))[:, 1]

    yv = y.to_numpy().astype(int)
    pc = np.clip(p, EPS, 1 - EPS)

    out = pd.DataFrame({
        "GAME_ID": meta["GAME_ID"].values,
        "GAME_DATE": pd.to_datetime(meta["GAME_DATE"]).values,
        "SEASON_ID": meta["SEASON_ID"].values,
        "season_label": meta["SEASON_ID"].map(SEASON_LABEL).values,
        "split": meta["SEASON_ID"].map(_split).values,
        "HOME_TEAM_ABBR": meta["HOME_TEAM_ABBR"].values,
        "HOME_TEAM_NAME": meta["HOME_TEAM_NAME"].values,
        "AWAY_TEAM_ABBR": meta["AWAY_TEAM_ABBR"].values,
        "AWAY_TEAM_NAME": meta["AWAY_TEAM_NAME"].values,
        "home_win_prob": p,
        "away_win_prob": 1 - p,
        "predicted_home_win": (p >= 0.5).astype(int),
        "actual_home_win": yv,
        "home_margin": y_margin.to_numpy(),
        "pred_margin": pred_margin,
        "margin_win_prob": margin_win_prob,
        "HOME_elo": X["HOME_elo"].values,
        "AWAY_elo": X["AWAY_elo"].values,
        "elo_diff": X["elo_diff"].values,
    })
    out["correct"] = (out["predicted_home_win"] == out["actual_home_win"]).astype(int)
    out["abs_error"] = (out["home_win_prob"] - out["actual_home_win"]).abs()
    out["logloss_contrib"] = -(yv * np.log(pc) + (1 - yv) * np.log(1 - pc))
    out["brier_contrib"] = (p - yv) ** 2
    out["confidence"] = (out["home_win_prob"] - 0.5).abs() * 2
    edges = np.round(np.arange(0, 1.0001, 0.1), 2)
    labels = [f"{int(edges[i] * 100)}-{int(edges[i + 1] * 100)}%" for i in range(len(edges) - 1)]
    out["prob_bucket"] = pd.cut(out["home_win_prob"], bins=edges, labels=labels,
                                include_lowest=True)
    out["upset_flag"] = ((out["predicted_home_win"] != out["actual_home_win"]) &
                         ((out["home_win_prob"] - 0.5).abs() > 0.15)).astype(int)

    pred_path = os.path.join(OUT_DIR, "predictions.csv")
    out.to_csv(pred_path, index=False)

    # ---- team Elo history (long format) --------------------------------------
    base = meta[["GAME_DATE", "SEASON_ID"]].copy()
    base["GAME_DATE"] = pd.to_datetime(base["GAME_DATE"])
    home = base.assign(team=meta["HOME_TEAM_ABBR"].values,
                       elo_before=X["HOME_elo"].values, is_home=1)
    away = base.assign(team=meta["AWAY_TEAM_ABBR"].values,
                       elo_before=X["AWAY_elo"].values, is_home=0)
    elo = pd.concat([home, away], ignore_index=True)
    elo["season_label"] = elo["SEASON_ID"].map(SEASON_LABEL)
    elo = elo.sort_values(["team", "GAME_DATE"]).reset_index(drop=True)
    elo_path = os.path.join(OUT_DIR, "team_elo_history.csv")
    elo.to_csv(elo_path, index=False)

    # ---- calibration table --------------------------------------------------
    cal = (out.groupby(["split", "prob_bucket"], observed=True)
              .agg(n=("actual_home_win", "size"),
                   mean_pred=("home_win_prob", "mean"),
                   mean_actual=("actual_home_win", "mean"))
              .reset_index())
    cal_path = os.path.join(OUT_DIR, "calibration.csv")
    cal.to_csv(cal_path, index=False)

    print(f"wrote:\n  {pred_path}  ({len(out)} rows)")
    print(f"  {elo_path}  ({len(elo)} rows)")
    print(f"  {cal_path}  ({len(cal)} rows)")
    print("\nper-split metrics:")
    for s in ["train", "val", "test"]:
        g = out[out["split"] == s]
        print(f"  {s:6s} n={len(g):5d}  log_loss={g['logloss_contrib'].mean():.4f}  "
              f"acc={g['correct'].mean():.4f}  brier={g['brier_contrib'].mean():.4f}")


if __name__ == "__main__":
    main()

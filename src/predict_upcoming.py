"""Score scheduled games and append them to data/processed/upcoming_predictions.csv.

  python src/predict_upcoming.py

This is the only honest measurement the project has once the model is retrained
on every finished season: at that point no season is out-of-sample any more, and
the live log is the only set of predictions the model has not seen the answers to.

That is why every row carries predicted_at and model_version. A prediction is a
historical fact produced by a specific model at a specific moment; without both
stamps the track record cannot be attributed and is worth nothing. Rows are
appended, never overwritten - re-running the same day refreshes today's rows and
leaves the history alone.

Filter live metrics to stale_games == 0 (see feature_engineering.add_stale_games).
"""
import datetime as dt
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
OUT_PATH = os.path.join(ROOT, "data", "processed", "upcoming_predictions.csv")

# Production model for the BACKTEST (predict.py, dashboard pages 1-5). Must
# keep a held-out season. The model used to FORECAST upcoming games is chosen
# separately - see main() - and is currently v2.
KEY = ["GAME_ID", "model_version"]
EPS = 1e-6


def blend_proba(bundle, X):
    f = bundle["features"]
    p_lr = bundle["logreg"].predict_proba(
        bundle["scaler"].transform(X[f].fillna(bundle["medians"])))[:, 1]
    p_xgb = bundle["xgb"].predict_proba(X[f])[:, 1]
    return bundle["blend_w"] * p_lr + (1 - bundle["blend_w"]) * p_xgb


def score_upcoming(bundle, data_path=DATA_PATH, now=None):
    X, _, _, meta = load_and_split_data(data_path, include_future=True)
    mask = (meta["is_future"] == 1).to_numpy()
    if not mask.any():
        return pd.DataFrame()

    # The last played game the features rest on. stale_games only counts
    # unresolved games *inside the fetched window*; it cannot know about games
    # that exist in the schedule but were never fetched. The daily job never
    # hits that (its window starts today), but nothing enforces it, so every
    # row records the history it was built from and says so out loud.
    played = meta.loc[~mask, "GAME_DATE"]
    data_through = pd.to_datetime(played).max().date().isoformat() if len(played) else ""

    X, meta = X[mask], meta[mask]
    f = bundle["features"]

    p = blend_proba(bundle, X)
    pred_margin = bundle["xgb_margin"].predict(X[f])
    margin_win_prob = bundle["margin_link"].predict_proba(
        pred_margin.reshape(-1, 1))[:, 1]

    stamp = (now or dt.datetime.now(dt.timezone.utc)).isoformat(timespec="seconds")

    return pd.DataFrame({
        "GAME_ID": meta["GAME_ID"].values,
        "GAME_DATE": meta["GAME_DATE"].values,
        "SEASON_ID": meta["SEASON_ID"].values,
        "HOME_TEAM_ABBR": meta["HOME_TEAM_ABBR"].values,
        "AWAY_TEAM_ABBR": meta["AWAY_TEAM_ABBR"].values,
        "home_win_prob": p,
        "away_win_prob": 1 - p,
        "predicted_home_win": (p > 0.5).astype(int),
        "pred_margin": pred_margin,
        "margin_win_prob": margin_win_prob,
        "HOME_elo": X["HOME_elo"].values if "HOME_elo" in X else np.nan,
        "AWAY_elo": X["AWAY_elo"].values if "AWAY_elo" in X else np.nan,
        "elo_diff": X["elo_diff"].values if "elo_diff" in X else np.nan,
        "confidence": np.abs(p - 0.5) * 2,
        "stale_games": meta["stale_games"].values,
        "is_early_season": X["is_early_season"].values,
        "model_version": bundle.get("meta", {}).get("version", "unknown"),
        "data_through": data_through,
        "predicted_at": stamp,
    })


def backfill_results(log, data_path=DATA_PATH):
    """Fill in what actually happened, for predictions whose game has been played.

    A prediction is only evidence once its result is known, and the result
    arrives days after the prediction - via fetch_data, into games_final, long
    after the game left the schedule. Without this join the live log can show
    what the model expected but never what it got, so no live metric can be
    computed at all.

    Only rows with stale_games == 0 belong in a headline metric; the column is
    kept on every row so the dashboard can filter rather than guess.
    """
    played = pd.read_csv(data_path, dtype={"GAME_ID": str},
                         usecols=["GAME_ID", "is_future", "home_win", "home_margin"])
    played = played[played["is_future"] == 0]
    truth = dict(zip(played["GAME_ID"], played["home_win"]))
    margins = dict(zip(played["GAME_ID"], played["home_margin"]))

    log = log.copy()
    # Nullable Int64, not float: these are whole numbers that are simply absent
    # until the game is played. Float would write "1.0" / "26.0" and break any
    # consumer typing them as whole numbers.
    log["actual_home_win"] = log["GAME_ID"].map(truth).astype("Int64")
    log["home_margin"] = log["GAME_ID"].map(margins).astype("Int64")

    known = log["actual_home_win"].notna()
    p = log["home_win_prob"].clip(EPS, 1 - EPS)
    y = log["actual_home_win"]

    log["correct"] = np.where(known, (log["predicted_home_win"] == y), np.nan)
    log["logloss_contrib"] = np.where(
        known, -(y * np.log(p) + (1 - y) * np.log(1 - p)), np.nan)
    log["brier_contrib"] = np.where(known, (p - y) ** 2, np.nan)
    return log


def merge_into_log(new_rows, out_path=OUT_PATH):
    """Append, keeping the newest prediction per (game, model version)."""
    if os.path.exists(out_path):
        old = pd.read_csv(out_path, dtype={"GAME_ID": str})
        combined = pd.concat([old, new_rows], ignore_index=True)
    else:
        combined = new_rows.copy()

    before = len(combined)
    combined = (combined.sort_values("predicted_at")
                        .drop_duplicates(subset=KEY, keep="last")
                        .sort_values(["GAME_DATE", "GAME_ID"])
                        .reset_index(drop=True))
    return combined, before - len(combined)


def main(model_paths=None):
    """Score fixtures with one or more models.

    Passing several models runs them in shadow: they all predict the same
    unplayed games, the log keys on (GAME_ID, model_version) so they sit side
    by side, and the season settles which is better on data none of them saw.
    Only the model at MODEL_PATH is production - it is the one predict.py uses
    for pages 1-5, and the one whose backtest is still honest.
    """
    model_paths = model_paths or [MODEL_PATH]
    bundles = [joblib.load(p) for p in model_paths]
    version = ", ".join(b.get("meta", {}).get("version", "unknown") for b in bundles)

    scored = []
    for i, b in enumerate(bundles):
        df = score_upcoming(b)
        # The FIRST model listed is primary - the one the dashboard shows by
        # default. It is deliberately not the same model predict.py uses for
        # pages 1-5, and that is not an inconsistency: the two answer different
        # questions. The backtest needs a model with a held-out season or its
        # numbers mean nothing. A forecast has no such constraint - the season
        # ahead is unseen by every candidate - so it should simply use the model
        # fit on the most and the most recent data.
        df["role"] = "primary" if i == 0 else "shadow"
        scored.append(df)
    scored = [df for df in scored if not df.empty]
    new_rows = pd.concat(scored, ignore_index=True) if scored else pd.DataFrame()
    if new_rows.empty:
        print("No scheduled games in games_final.csv - "
              "run fetch_schedule.py, then clean_data.py and feature_engineering.py")
        return

    combined, refreshed = merge_into_log(new_rows)
    combined = backfill_results(combined)
    combined.to_csv(OUT_PATH, index=False)

    settled = combined[combined["actual_home_win"].notna() &
                       (combined["stale_games"] == 0)]
    if len(settled):
        print("live track record (settled games, exact state):")
        for (ver, role), g in settled.groupby(["model_version", "role"]):
            print(f"  {ver} ({role}): n={len(g)}  "
                  f"log loss {g['logloss_contrib'].mean():.4f}  "
                  f"acc {g['correct'].mean():.3f}")

    exact = new_rows[new_rows["stale_games"] == 0]
    through = new_rows["data_through"].iloc[0]
    print(f"model {version}: scored {len(new_rows)} scheduled games "
          f"({len(exact)} with exact state, {len(new_rows) - len(exact)} stale)")
    print(f"features built from games played through {through}")
    print(f"log now holds {len(combined)} rows ({refreshed} refreshed) -> {OUT_PATH}\n")

    show = ["GAME_DATE", "HOME_TEAM_ABBR", "AWAY_TEAM_ABBR", "home_win_prob",
            "pred_margin", "stale_games"]
    print(new_rows[show].to_string(index=False,
                                   formatters={"home_win_prob": "{:.3f}".format,
                                               "pred_margin": "{:+.1f}".format}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="model files to score with (default: the production "
                             "copy). Pass several to run shadow models alongside.")
    args = parser.parse_args()
    main(model_paths=args.models)

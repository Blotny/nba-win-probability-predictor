"""Fit the production win-probability model and save it to models/.

The model is a 50/50 blend of a logistic regression and an XGBoost classifier,
trained on 39 selected features over the train+val seasons (2020-21 .. 2024-25).
A point-margin regressor is bundled alongside as a secondary estimate. The
2025-26 season is held out and only used for the sanity report at the end.

Run from anywhere:  python src/train_final_model.py
"""
import argparse
import os
import sys
import datetime as dt

import numpy as np
import joblib
import sklearn
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier, XGBRegressor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.train_model import load_and_split_data, split_by_season  # noqa: E402

DATA_PATH = os.path.join(ROOT, "data", "processed", "games_final.csv")
MODEL_PATH = os.path.join(ROOT, "models", "win_prob_blend.joblib")
MODEL_DIR = os.path.dirname(MODEL_PATH)

# Bump on every retrain that changes what the model learned (new seasons, new
# features, new hyper-parameters). Once predictions are made live, a prediction
# is a historical fact produced by a specific model - overwriting the joblib
# without a version makes the live track record unattributable and therefore
# worthless. win_prob_blend.joblib stays the production copy; the versioned
# file next to it is the archive.
MODEL_VERSION = "v1"

RANDOM_STATE = 42
BLEND_W = 0.5                     # weight on the logistic regression
LOGREG_C = 0.02
XGB_PARAMS = dict(max_depth=2, learning_rate=0.02, subsample=0.7,
                  colsample_bytree=0.7, min_child_weight=1,
                  reg_lambda=1.0, reg_alpha=0.0)
XGB_N_ESTIMATORS = 284           # from early stopping in notebooks/model_training.ipynb

TRAIN_SEASONS = [22020, 22021, 22022, 22023]
VAL_SEASONS = [22024]
TEST_SEASONS = [22025]

# 39 features chosen by L1-logreg + XGBoost importance in
# notebooks/model_training.ipynb section 6. Regenerate if feature engineering changes.
SELECTED = [
    "AWAY_elo", "AWAY_games_last_3d", "AWAY_games_last_7d", "AWAY_is_back_to_back",
    "AWAY_matchup_fg3_pct_5", "AWAY_matchup_pts_5", "AWAY_matchup_reb_5",
    "AWAY_matchup_stl_5", "AWAY_rest_days", "AWAY_rolling_stl_5", "AWAY_rolling_tov_10",
    "AWAY_rolling_tov_5", "AWAY_three_in_four", "HOME_elo", "HOME_games_last_3d",
    "HOME_games_played_this_season", "HOME_is_back_to_back", "HOME_matchup_ast_5",
    "HOME_matchup_blk_5", "HOME_matchup_fg3_pct_5", "HOME_matchup_reb_5",
    "HOME_rest_days", "HOME_rolling_blk_10", "HOME_rolling_off_rating_10",
    "HOME_rolling_poss_10", "HOME_rolling_stl_10", "HOME_rolling_tov_5",
    "HOME_rolling_venue_net_rating_10", "def_rating_diff_10", "elo_diff",
    "elo_win_prob", "games_last_3d_diff", "matchup_history_count",
    "net_rating_adj_diff_10", "off_rating_diff_10", "poss_diff_10", "rest_days_diff",
    "venue_net_rating_diff_10", "venue_net_rating_diff_5",
]


def fit_logreg(X, y, features):
    medians = X[features].median()
    scaler = StandardScaler().fit(X[features].fillna(medians))
    model = LogisticRegression(max_iter=2000, C=LOGREG_C).fit(
        scaler.transform(X[features].fillna(medians)), y)
    return model, scaler, medians


def fit_xgb_classifier(X, y, features):
    model = XGBClassifier(n_estimators=XGB_N_ESTIMATORS, eval_metric="logloss",
                          random_state=RANDOM_STATE, **XGB_PARAMS)
    model.fit(X[features], y, verbose=False)
    return model


def blend_proba(bundle, X):
    f = bundle["features"]
    p_lr = bundle["logreg"].predict_proba(
        bundle["scaler"].transform(X[f].fillna(bundle["medians"])))[:, 1]
    p_xgb = bundle["xgb"].predict_proba(X[f])[:, 1]
    return bundle["blend_w"] * p_lr + (1 - bundle["blend_w"]) * p_xgb


def _report(tag, y, p):
    print(f"  {tag:10s} n={len(y):5d}  log_loss={log_loss(y, p):.4f}  "
          f"acc={accuracy_score(y, p >= 0.5):.4f}  auc={roc_auc_score(y, p):.4f}  "
          f"brier={brier_score_loss(y, p):.4f}")


def main(version=MODEL_VERSION, fit_all=False, promote=True):
    """Fit the production blend and save it.

    fit_all=True fits on every season including the holdout. That model has no
    honest backtest left - by construction nothing is out-of-sample - so it must
    not replace the production copy (promote=False). Its evaluation is the live
    season instead: predict_upcoming.py scores fixtures with several models at
    once and upcoming_predictions.csv is keyed on (GAME_ID, model_version), so
    two versions predict the same unplayed games side by side and settle it on
    data neither has seen.
    """
    X, y, y_margin, metadata = load_and_split_data(DATA_PATH)
    split_by_season(X, y, metadata, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS)

    missing = [c for c in SELECTED if c not in X.columns]
    if missing:
        raise SystemExit(f"selected features missing from games_final.csv: {missing}")

    fit_seasons = (TRAIN_SEASONS + VAL_SEASONS + TEST_SEASONS if fit_all
                   else TRAIN_SEASONS + VAL_SEASONS)
    fit_mask = metadata["SEASON_ID"].isin(fit_seasons)
    test_mask = metadata["SEASON_ID"].isin(TEST_SEASONS)
    X_fit, y_fit, ymarg_fit = X[fit_mask], y[fit_mask], y_margin[fit_mask]

    logreg, scaler, medians = fit_logreg(X_fit, y_fit, SELECTED)
    xgb = fit_xgb_classifier(X_fit, y_fit, SELECTED)

    xgb_margin = XGBRegressor(n_estimators=XGB_N_ESTIMATORS, eval_metric="rmse",
                              random_state=RANDOM_STATE, **XGB_PARAMS)
    xgb_margin.fit(X_fit[SELECTED], ymarg_fit, verbose=False)
    pm_fit = xgb_margin.predict(X_fit[SELECTED])
    margin_link = LogisticRegression().fit(pm_fit.reshape(-1, 1), y_fit)

    bundle = dict(
        model_type="blend(logreg+xgb) on 39 selected features",
        features=SELECTED,
        blend_w=BLEND_W,
        logreg=logreg, scaler=scaler, medians=medians,
        xgb=xgb,
        xgb_margin=xgb_margin, margin_link=margin_link,
        meta=dict(
            version=version,
            trained_seasons=fit_seasons,
            val_seasons=[],  # the production fit uses train+val; nothing is held back
            test_season=[] if fit_all else TEST_SEASONS,
            created_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            sklearn=sklearn.__version__,
            xgboost=xgboost.__version__,
            n_features=len(SELECTED),
        ),
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    stamp = dt.date.today().isoformat()
    archive = os.path.join(MODEL_DIR, f"win_prob_blend_{version}_{stamp}.joblib")
    joblib.dump(bundle, archive)
    print(f"archived -> {archive}")

    if promote:
        joblib.dump(bundle, MODEL_PATH)
        print(f"promoted -> {MODEL_PATH}")
    else:
        print(f"NOT promoted: {os.path.basename(MODEL_PATH)} left untouched")

    print("\nsanity check (blend win-probability):")
    _report("fit", y_fit, blend_proba(bundle, X_fit))
    if fit_all:
        print("  (no held-out season - this model saw everything. "
              "Its real evaluation is the live season.)")
    else:
        _report("test", y[test_mask], blend_proba(bundle, X[test_mask]))
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default=MODEL_VERSION,
                        help=f"version tag (default {MODEL_VERSION})")
    parser.add_argument("--fit-all", action="store_true",
                        help="fit on every season including the holdout; implies "
                             "no promotion unless --promote is passed")
    parser.add_argument("--promote", action="store_true", default=None,
                        help="overwrite win_prob_blend.joblib (the production copy)")
    args = parser.parse_args()
    promote = (not args.fit_all) if args.promote is None else args.promote
    main(version=args.version, fit_all=args.fit_all, promote=promote)

"""Prove that features computed for a scheduled game match what they will be
once the game is played.

The whole upcoming-games path rests on one assumption: a row flagged
is_future=1 gets the same feature values as the same game computed after the
fact. If that silently breaks, the model is scored on garbage and nothing
raises an error - it just predicts badly, months later, for no visible reason.

This tests it offline, with no schedule API and no live games, by replaying the
end of a finished season as if it were the future: take the last N days of
2025-26, blank their box scores, run the real pipeline, and compare.

  python tests/test_future_features.py

Two scenarios:

  horizon 1 day  - every team's history is complete, so features MUST match
                   exactly. This is the pass/fail check.
  horizon 7 days - teams play inside the window, so games later in it are
                   predicted from stale state. This measures how fast that
                   drift grows, which is what decides how far ahead it is
                   honest to predict.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import src.clean_data as clean_data          # noqa: E402
import src.feature_engineering as fe         # noqa: E402

RAW_PATH = os.path.join(ROOT, 'data', 'raw', 'games.csv')
TRUTH_PATH = os.path.join(ROOT, 'data', 'processed', 'games_final.csv')
MODEL_PATH = os.path.join(ROOT, 'models', 'win_prob_blend.joblib')

# everything a scheduled game does not have yet
BOX_SCORE_COLS = ['WL', 'MIN', 'PTS', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                  'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB',
                  'AST', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']

NON_FEATURE_COLS = ['GAME_ID', 'GAME_DATE', 'SEASON_ID',
                    'HOME_TEAM_ID', 'HOME_TEAM_ABBR', 'HOME_TEAM_NAME',
                    'AWAY_TEAM_ID', 'AWAY_TEAM_ABBR', 'AWAY_TEAM_NAME',
                    'home_win', 'home_margin', 'is_future', 'stale_games']

TOL = 1e-9


def build_as_if_future(horizon_days, workdir):
    """Blank the last `horizon_days` of the raw file and run the real pipeline."""
    raw = pd.read_csv(RAW_PATH, dtype={'GAME_ID': str, 'SEASON_ID': str})
    raw['GAME_DATE'] = pd.to_datetime(raw['GAME_DATE'])

    cutoff = raw['GAME_DATE'].max() - pd.Timedelta(days=horizon_days - 1)
    future_mask = raw['GAME_DATE'] >= cutoff
    raw.loc[future_mask, BOX_SCORE_COLS] = np.nan

    raw_path = os.path.join(workdir, 'games.csv')
    raw.to_csv(raw_path, index=False)

    # point the pipeline modules at the sandbox instead of data/
    clean_data.INPUT_PATH = raw_path
    # isolate from the real schedule file, or live fixtures leak into the replay
    clean_data.SCHEDULE_PATH = os.path.join(workdir, 'no_schedule.csv')
    clean_data.UNMATCHED_OUTPUT_PATH = os.path.join(workdir, 'games_clean.csv')
    clean_data.MATCHED_OUTPUT_PATH = os.path.join(workdir, 'games_matched.csv')
    fe.INPUT_CLEAN_PATH = clean_data.UNMATCHED_OUTPUT_PATH
    fe.INPUT_MATCHED_PATH = clean_data.MATCHED_OUTPUT_PATH
    fe.OUTPUT_PATH = os.path.join(workdir, 'games_final.csv')

    clean_data.main()
    fe.main()

    out = pd.read_csv(fe.OUTPUT_PATH)
    return out, cutoff


def blend_proba(bundle, X):
    f = bundle['features']
    p_lr = bundle['logreg'].predict_proba(
        bundle['scaler'].transform(X[f].fillna(bundle['medians'])))[:, 1]
    p_xgb = bundle['xgb'].predict_proba(X[f])[:, 1]
    return bundle['blend_w'] * p_lr + (1 - bundle['blend_w']) * p_xgb


def compare(horizon_days):
    workdir = tempfile.mkdtemp(prefix='nba_future_')
    try:
        got, cutoff = build_as_if_future(horizon_days, workdir)
    finally:
        pass  # workdir removed by caller path below

    truth = pd.read_csv(TRUTH_PATH)
    for df in (got, truth):
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    scored = got[got['is_future'] == 1].set_index('GAME_ID').sort_index()
    ref = truth[truth['GAME_ID'].isin(scored.index)].set_index('GAME_ID').sort_index()

    assert scored.index.equals(ref.index), 'game sets differ'

    feat_cols = [c for c in truth.columns
                 if c not in NON_FEATURE_COLS and c in scored.columns]
    numeric = [c for c in feat_cols
               if pd.api.types.is_numeric_dtype(truth[c])]

    print(f'\n{"=" * 68}')
    print(f'HORIZON {horizon_days} day(s)  |  cutoff {cutoff.date()}  |  '
          f'{len(scored)} games  |  {len(numeric)} features')
    print('=' * 68)

    a = scored[numeric].astype(float)
    b = ref[numeric].astype(float)
    both_nan = a.isna() & b.isna()
    diff = (a - b).abs().mask(both_nan, 0.0)
    nan_mismatch = a.isna() ^ b.isna()

    per_col = diff.max()
    bad_cols = per_col[per_col > TOL]
    print(f'features differing        : {len(bad_cols)} / {len(numeric)}')
    print(f'NaN presence mismatches   : {int(nan_mismatch.values.sum())} cells')
    if len(bad_cols):
        print('\n  worst columns:')
        for col, v in bad_cols.sort_values(ascending=False).head(8).items():
            print(f'    {col:<34} max |diff| = {v:.4f}')

    # per-day breakdown: drift should be zero on day 1 and grow after
    scored_dates = scored['GAME_DATE']
    rows_bad = (diff > TOL).any(axis=1)
    print('\n  by game date:')
    for d, grp in scored.groupby(scored_dates):
        n_bad = int(rows_bad.loc[grp.index].sum())
        print(f'    {d.date()}  {len(grp):>2} games   {n_bad:>2} with drifted features')

    # what the drift costs in actual predictions
    if os.path.exists(MODEL_PATH):
        import joblib
        bundle = joblib.load(MODEL_PATH)
        p_future = blend_proba(bundle, scored)
        p_truth = blend_proba(bundle, ref)
        dp = np.abs(p_future - p_truth)
        print(f'\n  win-probability shift     : mean {dp.mean():.5f}  '
              f'max {dp.max():.5f}')

        # How bad is each staleness level, really? The dashboard bands are
        # based on these numbers, not on a guess about what "stale" ought to
        # cost. 'flips' is what matters to a reader: how often the stale state
        # points at the other team.
        if 'stale_games' in scored.columns:
            by = pd.DataFrame({'stale': scored['stale_games'].values, 'dp': dp,
                               'flip': (p_future > 0.5) != (p_truth > 0.5)})
            print()
            print('  drift by stale_games:')
            print(f'    {"stale":>5} {"games":>6} {"mean":>8} {"max":>8} {"flips":>7}')
            for lvl, g in by.groupby('stale'):
                print(f'    {lvl:>5} {len(g):>6} {g.dp.mean():>8.4f} '
                      f'{g.dp.max():>8.4f} {int(g.flip.sum()):>7}')

    shutil.rmtree(workdir, ignore_errors=True)
    return len(bad_cols), int(nan_mismatch.values.sum())


def main():
    bad_1, nan_1 = compare(1)
    compare(7)

    print(f'\n{"=" * 68}')
    if bad_1 == 0 and nan_1 == 0:
        print('PASS - at a 1-day horizon, scheduled-game features are identical '
              'to\n       the same games computed after they were played.')
        return 0
    print(f'FAIL - {bad_1} feature(s) and {nan_1} NaN cell(s) differ at a 1-day '
          'horizon.')
    return 1


if __name__ == '__main__':
    sys.exit(main())

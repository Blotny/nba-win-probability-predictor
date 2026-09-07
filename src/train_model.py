import pandas as pd

INPUT_PATH = 'data/processed/games_final.csv'

def load_and_split_data(input_path=INPUT_PATH, include_future=False):
    """Load games_final.csv into (X, y, y_margin, metadata).

    Scheduled games (is_future == 1) have no target and are dropped by default,
    so training code can never see them. predict_upcoming.py passes
    include_future=True to score them.
    """
    # GAME_ID is an identifier, not a number. Reading it as int makes it compare
    # unequal to the same id read as text elsewhere, which silently defeated the
    # de-duplication in predict_upcoming.merge_into_log.
    matched = pd.read_csv(input_path, dtype={'GAME_ID': str})

    if 'is_future' in matched.columns and not include_future:
        matched = matched[matched['is_future'] == 0].reset_index(drop=True)

    metadata_cols = [
        'GAME_ID', 'GAME_DATE', 'SEASON_ID',
        'HOME_TEAM_ID', 'HOME_TEAM_ABBR', 'HOME_TEAM_NAME',
        'AWAY_TEAM_ID', 'AWAY_TEAM_ABBR', 'AWAY_TEAM_NAME'
    ]
    target_col = 'home_win'
    # 'home_margin' is an outcome (home points - away points); it is a regression
    # target for the margin model, never a feature.
    target_cols = [target_col, 'home_margin']
    # 'is_future' is bookkeeping, not signal - it is constant across the training
    # set, so leaving it in X would hand the model a dead column (and a live one
    # the day fixtures are scored).
    excluded = metadata_cols + target_cols + ['is_future', 'stale_games']
    feature_cols = [c for c in matched.columns if c not in excluded]

    X = matched[feature_cols]
    y = matched[target_col]
    y_margin = matched['home_margin'] if 'home_margin' in matched.columns else None
    metadata = matched[[c for c in metadata_cols + ['is_future', 'stale_games']
                        if c in matched.columns]]

    return X, y, y_margin, metadata

# train test split
def split_by_season(X, y, metadata, train_seasons, val_seasons, test_seasons):
    mask_train = metadata['SEASON_ID'].isin(train_seasons)
    mask_val = metadata['SEASON_ID'].isin(val_seasons)
    mask_test = metadata['SEASON_ID'].isin(test_seasons)
 
    X_train, y_train, meta_train = X[mask_train], y[mask_train], metadata[mask_train]
    X_val, y_val, meta_val = X[mask_val], y[mask_val], metadata[mask_val]
    X_test, y_test, meta_test = X[mask_test], y[mask_test], metadata[mask_test]
 
    print(f"Train: {len(X_train)} matches ({train_seasons})")
    print(f"Val:   {len(X_val)} matches ({val_seasons})")
    print(f"Test:  {len(X_test)} matches ({test_seasons})")
 
    return X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test

if __name__ == '__main__':
    X, y, y_margin, metadata = load_and_split_data()

    TRAIN_SEASONS = [22020, 22021, 22022, 22023]
    VAL_SEASONS = [22024]
    TEST_SEASONS = [22025]

    X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = split_by_season(
        X, y, metadata, TRAIN_SEASONS, VAL_SEASONS, TEST_SEASONS
    )

    # baseline - home always wins
    baseline_accuracy = y_test.mean()
    print(f"Baseline: {baseline_accuracy:.3f}")
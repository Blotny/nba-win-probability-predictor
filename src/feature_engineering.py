import argparse

import numpy as np
import pandas as pd

INPUT_CLEAN_PATH = 'data/processed/games_clean.csv'
INPUT_MATCHED_PATH = 'data/processed/games_matched.csv'
OUTPUT_PATH = 'data/processed/games_final.csv'

ROLLING_WINDOWS = [5, 10]
MATCHUP_WINDOW = 5
EARLY_SEASON_GAMES = 10

# Elo hyper-parameters. Defaults below are the best combination found by
# tune_elo() on a rolling-origin split of seasons 22022/22023/22024
# (run `python src/feature_engineering.py --tune-elo` to reproduce the sweep).
ELO_K = 20
ELO_HOME_ADVANTAGE = 50
ELO_INITIAL = 1500
ELO_SEASON_REGRESSION = 0.6

# points of net rating credited/debited per 100 Elo points of opponent strength,
# used to build the opponent-adjusted efficiency features
OPP_ADJ_STRENGTH = 3.0

EFF_STAT_COLS = ['off_rating', 'def_rating', 'net_rating', 'net_rating_adj', 'poss']


def add_rest_days(df):
    df = df.sort_values(['TEAM_ABBREVIATION', 'GAME_DATE'])

    df['rest_days'] = df.groupby(['TEAM_ABBREVIATION', 'SEASON_ID'])['GAME_DATE'].diff().dt.days

    return df

def add_back_to_back(df):
    df['is_back_to_back'] = (df['rest_days'] == 1).astype(int)

    return df

def add_rolling_averages(df, window=5, stat_cols=None):
    if stat_cols is None:
        stat_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT']

    for col in stat_cols:
        df[f'rolling_{col.lower()}_{window}'] = df.groupby(['TEAM_ABBREVIATION', 'SEASON_ID'])[col].transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())

    return df


def add_games_played(df):
    # number of games the team has already played this season BEFORE the current one
    # (df is sorted by team / season / date in main())
    df['games_played_this_season'] = df.groupby(['TEAM_ABBREVIATION', 'SEASON_ID']).cumcount()

    return df


def _elo_expected_home(home_rating, away_rating, home_advantage):
    return 1 / (1 + 10 ** ((away_rating - (home_rating + home_advantage)) / 400))


def calculate_elo(matched_df, k=ELO_K, home_advantage=ELO_HOME_ADVANTAGE,
                  initial_elo=ELO_INITIAL, season_regression=ELO_SEASON_REGRESSION,
                  margin_col='home_margin'):
    # matched_df: one row one match, sorted by GAME_DATE.
    # Uses a FiveThirtyEight-style margin-of-victory multiplier so blowouts move
    # ratings more than one-possession games, damped for lopsided matchups.

    ratings = {}  # team abbr -> current rating
    home_elo_before = []
    away_elo_before = []
    elo_win_prob = []  # pre-game P(home win) from Elo alone
    current_season = None

    has_margin = margin_col in matched_df.columns

    for idx, row in matched_df.iterrows():
        team_home = row['HOME_TEAM_ABBR']
        team_away = row['AWAY_TEAM_ABBR']
        season = row['SEASON_ID']

        # 1. season changed -> regress every rating towards the mean
        if current_season is not None and season != current_season:
            for team in ratings:
                ratings[team] = season_regression * ratings[team] + (1 - season_regression) * initial_elo
        current_season = season

        # 2. load current ratings
        home_rating = ratings.get(team_home, initial_elo)
        away_rating = ratings.get(team_away, initial_elo)

        # 3. store pre-game state
        home_elo_before.append(home_rating)
        away_elo_before.append(away_rating)

        expected_home = _elo_expected_home(home_rating, away_rating, home_advantage)
        elo_win_prob.append(expected_home)

        actual_home = row['home_win']

        # 4. margin-of-victory multiplier (falls back to 1.0 without a margin col)
        if has_margin:
            margin = abs(row[margin_col])
            if actual_home == 1:
                elo_diff_w = (home_rating + home_advantage) - away_rating
            else:
                elo_diff_w = away_rating - (home_rating + home_advantage)
            mov_mult = ((margin + 3) ** 0.8) / (7.5 + 0.006 * elo_diff_w)
        else:
            mov_mult = 1.0

        # 5. symmetric update (sum of rating changes stays 0)
        change = k * mov_mult * (actual_home - expected_home)
        ratings[team_home] = home_rating + change
        ratings[team_away] = away_rating - change

    matched_df['HOME_elo'] = home_elo_before
    matched_df['AWAY_elo'] = away_elo_before
    matched_df['elo_win_prob'] = elo_win_prob
    return matched_df


def _prior_game_counts(matched):
    # games each team has already played THIS season before each row
    # (matched must be sorted by GAME_DATE)
    counts = {}
    home_n, away_n = [], []
    current_season = None
    for _, row in matched.iterrows():
        if row['SEASON_ID'] != current_season:
            counts = {}
            current_season = row['SEASON_ID']
        h, a = row['HOME_TEAM_ABBR'], row['AWAY_TEAM_ABBR']
        home_n.append(counts.get(h, 0))
        away_n.append(counts.get(a, 0))
        counts[h] = counts.get(h, 0) + 1
        counts[a] = counts.get(a, 0) + 1
    return np.array(home_n), np.array(away_n)


def tune_elo(matched, k_grid=(10, 20, 30, 40),
             home_adv_grid=(50, 75, 100, 125),
             regression_grid=(0.6, 0.75, 0.9, 1.0),
             val_seasons=(22022, 22023, 22024),
             warmup_games=EARLY_SEASON_GAMES):
    # Sweep Elo hyper-parameters, scoring a pure-Elo win-probability model by
    # log loss on the validation seasons (early-season games dropped).
    matched = matched.sort_values('GAME_DATE').reset_index(drop=True)
    home_n, away_n = _prior_game_counts(matched)
    warm = (home_n >= warmup_games) & (away_n >= warmup_games)
    val_mask = matched['SEASON_ID'].isin(val_seasons).values & warm
    y = matched['home_win'].values[val_mask]

    rows = []
    for k in k_grid:
        for home_adv in home_adv_grid:
            for reg in regression_grid:
                scored = calculate_elo(matched.copy(), k=k, home_advantage=home_adv,
                                       season_regression=reg)
                p = scored['elo_win_prob'].values[val_mask]
                p = np.clip(p, 1e-6, 1 - 1e-6)
                ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                acc = np.mean((p >= 0.5) == y)
                rows.append({'k': k, 'home_advantage': home_adv,
                             'season_regression': reg, 'val_log_loss': ll,
                             'val_acc': acc})

    return pd.DataFrame(rows).sort_values('val_log_loss').reset_index(drop=True)


def add_streak(df):
    df = df.sort_values(['TEAM_ABBREVIATION', 'SEASON_ID', 'GAME_DATE']).reset_index(drop=True)

    streak_before_game = []
    current_streak = 0
    current_team = 0
    current_season = None

    for idx, row in df.iterrows():
        team = row['TEAM_ABBREVIATION']
        season = row['SEASON_ID']
        if team != current_team or season != current_season:
            current_streak = 0

        streak_before_game.append(current_streak)

        if row['WL'] == 'W':
            if current_streak >= 0:
                current_streak += 1
            else:
                current_streak = 1
        elif row['WL'] == 'L':
            if current_streak <= 0:
                current_streak -= 1
            else:
                current_streak = -1

        current_team = team
        current_season = season

        pass

    df['streak'] = streak_before_game
    return df


def add_efficiency_features(games_clean, elo_long):
    # possession-based efficiency, per team per game, from the box score plus the
    # opponent's box score (self-join on GAME_ID). Opponent Elo (pre-game) is used
    # for a light opponent-strength adjustment of net rating.
    opp = games_clean[['GAME_ID', 'TEAM_ABBREVIATION', 'PTS', 'FGA', 'FTA',
                       'OREB', 'TOV']].rename(columns={
        'TEAM_ABBREVIATION': 'OPP_ABBR', 'PTS': 'OPP_PTS', 'FGA': 'OPP_FGA',
        'FTA': 'OPP_FTA', 'OREB': 'OPP_OREB', 'TOV': 'OPP_TOV'})

    df = games_clean.merge(opp, on='GAME_ID')
    df = df[df['TEAM_ABBREVIATION'] != df['OPP_ABBR']].copy()

    team_poss = df['FGA'] + 0.44 * df['FTA'] - df['OREB'] + df['TOV']
    opp_poss = df['OPP_FGA'] + 0.44 * df['OPP_FTA'] - df['OPP_OREB'] + df['OPP_TOV']
    df['poss'] = 0.5 * (team_poss + opp_poss)

    df['off_rating'] = 100 * df['PTS'] / df['poss']
    df['def_rating'] = 100 * df['OPP_PTS'] / df['poss']
    df['net_rating'] = df['off_rating'] - df['def_rating']

    # opponent-adjusted: reward net rating earned against strong opponents
    df = df.merge(elo_long.rename(columns={'TEAM_ABBREVIATION': 'OPP_ABBR',
                                           'team_elo': 'opp_elo'}),
                  on=['GAME_ID', 'OPP_ABBR'], how='left')
    df['net_rating_adj'] = df['net_rating'] + OPP_ADJ_STRENGTH * (df['opp_elo'] - ELO_INITIAL) / 100

    df['is_home_game'] = df['MATCHUP'].str.contains('vs.', regex=False)

    df = df.drop(columns=['OPP_ABBR', 'OPP_PTS', 'OPP_FGA', 'OPP_FTA',
                          'OPP_OREB', 'OPP_TOV', 'opp_elo'])
    return df


def add_efficiency_rolling(df, window):
    # overall rolling efficiency
    grp = df.groupby(['TEAM_ABBREVIATION', 'SEASON_ID'])
    for col in EFF_STAT_COLS:
        df[f'rolling_{col}_{window}'] = grp[col].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())

    # venue-specific net rating: home form for home games, road form for away
    for venue, flag in [('home', True), ('away', False)]:
        sub = df[df['is_home_game'] == flag]
        rolled = sub.groupby(['TEAM_ABBREVIATION', 'SEASON_ID'])['net_rating'].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        df.loc[sub.index, f'_venue_net_rating_{window}'] = rolled
    df[f'rolling_venue_net_rating_{window}'] = df[f'_venue_net_rating_{window}']
    df = df.drop(columns=[f'_venue_net_rating_{window}'])

    return df


def add_schedule_load(df):
    # schedule congestion: games in the trailing 7 / 3 days (strictly before the
    # current game), plus common back-to-back-density flags.
    df = df.sort_values(['TEAM_ABBREVIATION', 'SEASON_ID', 'GAME_DATE']).reset_index(drop=True)

    g7 = np.zeros(len(df), dtype=int)
    g3 = np.zeros(len(df), dtype=int)
    for _, grp in df.groupby(['TEAM_ABBREVIATION', 'SEASON_ID']):
        dates = grp['GAME_DATE'].values
        idx = grp.index.values
        for i in range(len(dates)):
            lo7 = np.searchsorted(dates, dates[i] - np.timedelta64(7, 'D'), side='left')
            lo3 = np.searchsorted(dates, dates[i] - np.timedelta64(3, 'D'), side='left')
            g7[idx[i]] = i - lo7
            g3[idx[i]] = i - lo3

    df['games_last_7d'] = g7
    df['games_last_3d'] = g3
    df['three_in_four'] = (g3 >= 2).astype(int)   # >=2 games in prior 3 days + today = 3-in-4
    df['four_in_six'] = (g7 >= 3).astype(int)
    return df


def add_matchup_features(matched, window=MATCHUP_WINDOW, stat_cols=None):
    # head-to-head features for each game, using only PRIOR meetings of the same
    # pair of teams (cross-season). matched must still hold the raw HOME_/AWAY_
    # stats and 'home_win' at this point (before merge_features_into_matched).
    if stat_cols is None:
        stat_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT']

    matched = matched.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)

    history = {}  # matchup key -> list of past meetings (home abbr + per-team stats + winner abbr)

    matchup_history_count = []
    matchup_streak = []
    home_matchup_stats = {col: [] for col in stat_cols}
    away_matchup_stats = {col: [] for col in stat_cols}

    for idx, row in matched.iterrows():
        home_team = row['HOME_TEAM_ABBR']
        away_team = row['AWAY_TEAM_ABBR']
        key = '_'.join(sorted([home_team, away_team]))

        past = history.get(key, [])
        recent = past[-window:]

        matchup_history_count.append(len(past))

        # rolling H2H averages from the current home / away team perspective
        for col in stat_cols:
            if recent:
                home_vals = [m['stats'][home_team][col] for m in recent]
                away_vals = [m['stats'][away_team][col] for m in recent]
                home_matchup_stats[col].append(sum(home_vals) / len(home_vals))
                away_matchup_stats[col].append(sum(away_vals) / len(away_vals))
            else:
                home_matchup_stats[col].append(float('nan'))
                away_matchup_stats[col].append(float('nan'))

        # signed H2H streak from the current home team perspective
        # (sign logic mirrors add_streak); AWAY side is just the negation
        streak = 0
        for m in reversed(recent):
            home_won = m['winner_abbr'] == home_team
            if home_won:
                if streak >= 0:
                    streak += 1
                else:
                    break
            else:
                if streak <= 0:
                    streak -= 1
                else:
                    break
        matchup_streak.append(streak)

        # record the current meeting AFTER computing features (no leakage)
        winner_abbr = home_team if row['home_win'] == 1 else away_team
        history.setdefault(key, []).append({
            'winner_abbr': winner_abbr,
            'stats': {
                home_team: {col: row[f'HOME_{col}'] for col in stat_cols},
                away_team: {col: row[f'AWAY_{col}'] for col in stat_cols},
            },
        })

    matched['matchup_history_count'] = matchup_history_count
    matched['matchup_streak'] = matchup_streak
    for col in stat_cols:
        matched[f'HOME_matchup_{col.lower()}_{window}'] = home_matchup_stats[col]
        matched[f'AWAY_matchup_{col.lower()}_{window}'] = away_matchup_stats[col]

    return matched


def add_diff_features(matched):
    matched['elo_diff'] = matched['HOME_elo'] - matched['AWAY_elo']
    matched['rest_days_diff'] = matched['HOME_rest_days'] - matched['AWAY_rest_days']
    matched['streak_diff'] = matched['HOME_streak'] - matched['AWAY_streak']
    matched['games_last_7d_diff'] = matched['HOME_games_last_7d'] - matched['AWAY_games_last_7d']
    matched['games_last_3d_diff'] = matched['HOME_games_last_3d'] - matched['AWAY_games_last_3d']

    for w in ROLLING_WINDOWS:
        for col in ['net_rating', 'net_rating_adj', 'off_rating', 'def_rating',
                    'poss', 'venue_net_rating']:
            h, a = f'HOME_rolling_{col}_{w}', f'AWAY_rolling_{col}_{w}'
            if h in matched.columns and a in matched.columns:
                matched[f'{col}_diff_{w}'] = matched[h] - matched[a]

    return matched


def add_early_season_flag(matched, threshold=EARLY_SEASON_GAMES):
    matched['is_early_season'] = (
        (matched['HOME_games_played_this_season'] < threshold)
        | (matched['AWAY_games_played_this_season'] < threshold)
    ).astype(int)

    return matched


def merge_features_into_matched(games_clean, matched):
    base_cols = ['rest_days', 'is_back_to_back', 'streak', 'games_played_this_season',
                 'games_last_7d', 'games_last_3d', 'three_in_four', 'four_in_six']
    feature_cols = base_cols + [c for c in games_clean.columns if c.startswith('rolling_')]

    # home
    home_features = games_clean[['GAME_ID', 'TEAM_ABBREVIATION'] + feature_cols].copy()
    home_features.columns = ['GAME_ID', 'TEAM_ABBREVIATION'] + [f'HOME_{c}' for c in feature_cols]

    matched = matched.merge(
        home_features,
        left_on=['GAME_ID', 'HOME_TEAM_ABBR'],
        right_on=['GAME_ID', 'TEAM_ABBREVIATION'],
        how='left'
    )
    matched = matched.drop(columns=['TEAM_ABBREVIATION'])

    # away
    away_features = games_clean[['GAME_ID', 'TEAM_ABBREVIATION'] + feature_cols].copy()
    away_features.columns = ['GAME_ID', 'TEAM_ABBREVIATION'] + [f'AWAY_{c}' for c in feature_cols]

    matched = matched.merge(
        away_features,
        left_on=['GAME_ID', 'AWAY_TEAM_ABBR'],
        right_on=['GAME_ID', 'TEAM_ABBREVIATION'],
        how='left'
    )
    matched = matched.drop(columns=['TEAM_ABBREVIATION'])

    raw_stat_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT',
                  'FG3_PCT', 'FT_PCT', 'OREB', 'DREB', 'PF', 'PLUS_MINUS', 'WL']

    cols_to_drop = [f'HOME_{c}' for c in raw_stat_cols] + [f'AWAY_{c}' for c in raw_stat_cols]
    matched = matched.drop(columns=cols_to_drop)

    return matched


def build_games_clean(games_clean, elo_long):
    games_clean = games_clean.sort_values(
        ['TEAM_ABBREVIATION', 'SEASON_ID', 'GAME_DATE']).reset_index(drop=True)

    games_clean = add_rest_days(games_clean)
    games_clean = add_back_to_back(games_clean)
    for window in ROLLING_WINDOWS:
        games_clean = add_rolling_averages(games_clean, window=window)
    games_clean = add_streak(games_clean)
    games_clean = add_games_played(games_clean)
    games_clean = add_schedule_load(games_clean)

    games_clean = add_efficiency_features(games_clean, elo_long)
    games_clean = games_clean.sort_values(
        ['TEAM_ABBREVIATION', 'SEASON_ID', 'GAME_DATE']).reset_index(drop=True)
    for window in ROLLING_WINDOWS:
        games_clean = add_efficiency_rolling(games_clean, window=window)

    return games_clean


def main(tune=False):
    matched = pd.read_csv(INPUT_MATCHED_PATH)
    matched['GAME_DATE'] = pd.to_datetime(matched['GAME_DATE'])
    matched = matched.sort_values('GAME_DATE').reset_index(drop=True)
    matched['home_margin'] = matched['HOME_PTS'] - matched['AWAY_PTS']

    if tune:
        table = tune_elo(matched)
        print("Elo hyper-parameter sweep (best 10 by validation log loss):")
        print(table.head(10).to_string(index=False))
        best = table.iloc[0]
        print(f"\nBest: k={best.k}, home_advantage={best.home_advantage}, "
              f"season_regression={best.season_regression} "
              f"(val log loss {best.val_log_loss:.4f})")
        return

    matched = calculate_elo(matched)

    elo_long = pd.concat([
        matched[['GAME_ID', 'HOME_TEAM_ABBR', 'HOME_elo']].rename(
            columns={'HOME_TEAM_ABBR': 'TEAM_ABBREVIATION', 'HOME_elo': 'team_elo'}),
        matched[['GAME_ID', 'AWAY_TEAM_ABBR', 'AWAY_elo']].rename(
            columns={'AWAY_TEAM_ABBR': 'TEAM_ABBREVIATION', 'AWAY_elo': 'team_elo'}),
    ], ignore_index=True)

    games_clean = pd.read_csv(INPUT_CLEAN_PATH)
    games_clean['GAME_DATE'] = pd.to_datetime(games_clean['GAME_DATE'])
    games_clean = build_games_clean(games_clean, elo_long)

    matched = add_matchup_features(matched)

    matched = merge_features_into_matched(games_clean, matched)
    matched = add_diff_features(matched)
    matched = add_early_season_flag(matched)

    matched.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved final file to {OUTPUT_PATH}: {len(matched)} rows")

    print(matched.shape)  # musi być (7230, N) - liczba wierszy się NIE mogła zmienić
    print(matched[['GAME_DATE', 'HOME_TEAM_ABBR', 'HOME_streak', 'streak_diff',
                   'elo_win_prob', 'net_rating_diff_10', 'HOME_games_last_7d',
                   'home_margin', 'is_early_season']].head(10))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tune-elo', action='store_true',
                        help='sweep Elo hyper-parameters and exit')
    args = parser.parse_args()
    main(tune=args.tune_elo)

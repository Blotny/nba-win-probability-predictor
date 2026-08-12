import pandas as pd

INPUT_CLEAN_PATH = 'data/processed/games_clean.csv'
INPUT_MATCHED_PATH = 'data/processed/games_matched.csv'
OUTPUT_PATH = 'data/processed/games_final.csv'

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
        df[f'rolling_{col.lower()}_{window}'] = df.groupby(['TEAM_ABBREVIATION', 'SEASON_ID'])[col].shift(1).rolling(window=window, min_periods=1).mean()

    return df


def calculate_elo(matched_df, k=20, home_advantage=100, initial_elo=1500, season_regression=0.75):
    # matched_df: one row one match 
 
    ratings = {}  # dict: name of the team : current ranking
    home_elo_before = []  # rating before match
    away_elo_before = []
    current_season = None
 
    for idx, row in matched_df.iterrows():
        team_home = row['HOME_TEAM_ABBR']
        team_away = row['AWAY_TEAM_ABBR']
        season = row['SEASON_ID']
 
        # 1. season changed 
        if current_season is not None and season != current_season:
            for team in ratings:
                ratings[team] = season_regression * ratings[team] + (1 - season_regression) * initial_elo
        current_season = season
 
        # 2. load current rating (initial 1500)
        home_rating = ratings.get(team_home, initial_elo)
        away_rating = ratings.get(team_away, initial_elo)
 
        # 3. save rating
        home_elo_before.append(home_rating)
        away_elo_before.append(away_rating)
 
        # 4. expected score (with home bonus)
        expected_home = 1 / (1 + 10 ** ((away_rating - (home_rating + home_advantage)) / 400))
 
        # 5. actual score
        actual_home = row['home_win']
 
        # 6. updating ratings after match
        new_home_rating = home_rating + k * (actual_home - expected_home)
        new_away_rating = away_rating + k * ((1 - actual_home) - (1 - expected_home))
 
        ratings[team_home] = new_home_rating
        ratings[team_away] = new_away_rating
 
    matched_df['HOME_elo'] = home_elo_before
    matched_df['AWAY_elo'] = away_elo_before
    return matched_df


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


def merge_features_into_matched(games_clean, matched):
    feature_cols = ['rest_days', 'is_back_to_back', 'streak'] + \
                    [c for c in games_clean.columns if c.startswith('rolling_')]

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


def main():
    games_clean = pd.read_csv(INPUT_CLEAN_PATH)
    games_clean['GAME_DATE'] = pd.to_datetime(games_clean['GAME_DATE'])
    games_clean = games_clean.sort_values(['TEAM_ABBREVIATION', 'SEASON_ID', 'GAME_DATE']).reset_index(drop=True)

    games_clean = add_rest_days(games_clean)
    games_clean = add_back_to_back(games_clean)
    games_clean = add_rolling_averages(games_clean)
    games_clean = add_streak(games_clean)

    matched = pd.read_csv(INPUT_MATCHED_PATH)
    matched['GAME_DATE'] = pd.to_datetime(matched['GAME_DATE'])
    matched = matched.sort_values('GAME_DATE').reset_index(drop=True)
    matched = calculate_elo(matched)

    matched = merge_features_into_matched(games_clean, matched)

    matched.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved final file to {OUTPUT_PATH}: {len(matched)} rows")

    print(matched.shape)  # musi być (7230, N) - liczba wierszy się NIE mogła zmienić
    print(matched[['GAME_DATE', 'HOME_TEAM_ABBR', 'HOME_rest_days', 'HOME_streak', 'HOME_elo']].head(10))

if __name__ == '__main__':
    main()
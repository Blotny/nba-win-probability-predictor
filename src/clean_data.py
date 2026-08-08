import pandas as pd

INPUT_PATH = 'data/raw/games.csv'
OUTPUT_PATH = 'data/processed/games_matched.csv'

def load_raw_games():
    df = pd.read_csv(INPUT_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def split_home_away(df):
    # our data now have two rows for one game (home and away)
    # we want to split away games from home games for merging them

    # 'vs' in MATCHUP means home '@' means away
    home = df[df['MATCHUP'].str.contains('vs.')].copy()
    away = df[df['MATCHUP'].str.contains('@')].copy()

    print(f"Home rows: {len(home)}, Away rows: {len(away)}")

    return home, away

def merge_home_away(home, away):

    stat_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT',
                     'FG3_PCT', 'FT_PCT', 'OREB', 'DREB', 'PF', 'PLUS_MINUS', 'WL']
    
    home_renamed = home[['GAME_ID', 'GAME_DATE', 'SEASON_ID', 'TEAM_ID',
                              'TEAM_ABBREVIATION', 'TEAM_NAME'] + stat_cols].copy()
    home_renamed.columns = ['GAME_ID', 'GAME_DATE', 'SEASON_ID', 'HOME_TEAM_ID',
                                 'HOME_TEAM_ABBR', 'HOME_TEAM_NAME'] + [f'HOME_{c}' for c in stat_cols]
     
    away_renamed = away[['GAME_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME'] + stat_cols].copy()
    away_renamed.columns = ['GAME_ID', 'AWAY_TEAM_ID', 'AWAY_TEAM_ABBR', 'AWAY_TEAM_NAME'] + [f'AWAY_{c}' for c in stat_cols]

    merged = home_renamed.merge(away_renamed, on='GAME_ID', how='inner')

    return merged

def add_target(df):
    df['home_win'] = (df['HOME_WL'] == 'W').astype(int)
    return df

def main():
    print("Loading raw data...")
    raw = load_raw_games()
    print(f"Loaded {len(raw)} rows")
 
    print("Spliting home/away...")
    home, away = split_home_away(raw)
 
    print("Merging home/away...")
    matched = merge_home_away(home, away)
    print(f"After merging: {len(matched)} matches")
 
    matched = add_target(matched)
 
    # sorting
    matched = matched.sort_values('GAME_DATE').reset_index(drop=True)

    matched.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(matched)} matches to {OUTPUT_PATH}")
 
    print("\nPreview:")
    print(matched[['GAME_DATE', 'HOME_TEAM_ABBR', 'AWAY_TEAM_ABBR', 'HOME_PTS', 'AWAY_PTS', 'home_win']].head())

 
if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd

import os

INPUT_PATH = 'data/raw/games.csv'
SCHEDULE_PATH = 'data/raw/schedule.csv'
UNMATCHED_OUTPUT_PATH = 'data/processed/games_clean.csv'
MATCHED_OUTPUT_PATH = 'data/processed/games_matched.csv'


def load_raw_games():
    """Played games, plus any scheduled games that have not been played yet.

    A game appears in schedule.csv from the day it is published and in
    games.csv from the day it is played, so for a window of a few days both
    files hold it. games.csv always wins: it has the box score, the schedule
    row has nothing. Dropping the loser here is what stops a played game from
    reappearing as a fixture and being predicted a second time.
    """
    df = pd.read_csv(INPUT_PATH)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    if not os.path.exists(SCHEDULE_PATH):
        return df

    sched = pd.read_csv(SCHEDULE_PATH)
    if sched.empty:
        return df
    sched["GAME_DATE"] = pd.to_datetime(sched["GAME_DATE"])

    unplayed = sched[~sched["GAME_ID"].isin(df["GAME_ID"])]
    print(f"Schedule: {len(sched) // 2} games, "
          f"{(len(sched) - len(unplayed)) // 2} already played, "
          f"{len(unplayed) // 2} still upcoming")

    return pd.concat([df, unplayed], ignore_index=True)


def fix_corrupted_matchup(df):
    # there are some anomalies in data where matchup column is corrupted
    # we fix this column for correct split in next step

    starts_with_own_team = df['TEAM_ABBREVIATION'] == df['MATCHUP'].str[:3]
    valid = df[starts_with_own_team].copy()
    corrupted = df[~starts_with_own_team].copy()

    partner_info = valid[['GAME_ID', 'TEAM_ABBREVIATION', 'MATCHUP']].rename(
        columns={'TEAM_ABBREVIATION': 'PARTNER_TEAM', 'MATCHUP': 'PARTNER_MATCHUP'}
    )

    corrupted = corrupted.merge(partner_info, on='GAME_ID', how='left')

    def build_correct_matchup(row):
        own_team = row['TEAM_ABBREVIATION']
        partner_team = row['PARTNER_TEAM']
        partner_matchup = row['PARTNER_MATCHUP']

        if 'vs.' in partner_matchup:
            # partner was home team
            return f"{own_team} @ {partner_team}"
        else:
            # partner was away team
            return f"{own_team} vs. {partner_team}"

    corrupted['MATCHUP'] = corrupted.apply(build_correct_matchup, axis=1)

    corrupted = corrupted.drop(columns=['PARTNER_TEAM', 'PARTNER_MATCHUP'])

    fixed_df = pd.concat([valid, corrupted], ignore_index=True)

    return fixed_df


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

def add_is_future(df, wl_col):
    """Flag rows that have no result yet.

    A scheduled game is defined by the absence of a box score, not by an
    external marker: fetch_schedule appends rows with NaN stats and this
    derives everything downstream from that. It also makes the backtest in
    tests/test_future_features.py faithful - blanking a played game's box
    score produces exactly the row shape a real fixture has.
    """
    df['is_future'] = df[wl_col].isna().astype(int)
    return df


def add_target(df):
    # NaN, not 0, for games without a result - a future game has no target,
    # and 0 would silently teach the model that every fixture is a home loss.
    df['home_win'] = np.where(df['HOME_WL'].isna(), np.nan,
                              (df['HOME_WL'] == 'W').astype(float))
    return df

def main():
    raw = load_raw_games()
    raw_fixed = fix_corrupted_matchup(raw)
    raw_fixed = raw_fixed.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)

    raw_fixed = add_is_future(raw_fixed, 'WL')
    raw_fixed.to_csv(UNMATCHED_OUTPUT_PATH, index=False)
    print(f"Saved {len(raw_fixed)} rows to {UNMATCHED_OUTPUT_PATH}")

    home, away = split_home_away(raw_fixed)
    matched = merge_home_away(home, away)
    matched = add_is_future(matched, 'HOME_WL')
    matched = add_target(matched)
    matched = matched.sort_values('GAME_DATE').reset_index(drop=True)
    matched.to_csv(MATCHED_OUTPUT_PATH, index=False)
    print(f"Saved {len(matched)} matches to {MATCHED_OUTPUT_PATH}")
 
    print("\nPreview:")
    print(matched[['GAME_DATE', 'HOME_TEAM_ABBR', 'AWAY_TEAM_ABBR', 'HOME_PTS', 'AWAY_PTS', 'home_win']].head())
    print("Scheduled (is_future=1):", int(matched["is_future"].sum()), "games")

 
if __name__ == '__main__':
    main()
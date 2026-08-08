import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# 2020-24 - train   2024-25 - val   2025-26 - test
SEASONS = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26']

OUTPUT_PATH = 'data/raw/games.csv'

all_games = []

for season in SEASONS:
    print(f"Downloading season {season}...")
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable='00',
        season_type_nullable='Regular Season'
    )
    df = gamefinder.get_data_frames()[0]
    all_games.append(df)
    time.sleep(0.6)

games_df = pd.concat(all_games, ignore_index=True)
games_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(games_df)} rows to {OUTPUT_PATH}")
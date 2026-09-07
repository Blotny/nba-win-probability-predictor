"""Fetch raw NBA game data from nba_api into data/raw/games.csv.

Two modes:

  python src/fetch_data.py                # full rebuild of every season in SEASONS
  python src/fetch_data.py --incremental  # refetch CURRENT_SEASON only, splice into the file

Incremental is what the daily job runs. It re-pulls the *whole* current season
rather than only the newest games: the NBA revises box scores for a few days
after a game (assists reassigned, rebounds reclassified), so pulling only new
rows would let the file drift out of sync. Finished seasons never change and
are left untouched.
"""

import argparse
import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder

# 2020-24 - train   2024-25 - val   2025-26 - test
SEASONS = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25', '2025-26',
           '2026-27']
CURRENT_SEASON = SEASONS[-1]

OUTPUT_PATH = 'data/raw/games.csv'

REQUEST_PAUSE = 0.6  # nba_api is rate limited


def fetch_season(season):
    """Return one season of regular-season team-game rows (empty frame if not started)."""
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable='00',
        season_type_nullable='Regular Season'
    )
    return gamefinder.get_data_frames()[0]


def load_existing():
    """Read games.csv as text so untouched seasons round-trip unchanged.

    GAME_ID carries leading zeros ('0022001080') that an int cast would eat.
    """
    return pd.read_csv(OUTPUT_PATH, dtype=str)


def fetch_full():
    frames = []
    for season in SEASONS:
        print(f"Downloading season {season}...")
        df = fetch_season(season)
        if df.empty:
            print(f"  {season}: no games yet, skipping")
        else:
            print(f"  {season}: {len(df)} rows")
            frames.append(df)
        time.sleep(REQUEST_PAUSE)
    return pd.concat(frames, ignore_index=True)


def fetch_incremental():
    """Replace CURRENT_SEASON's rows in the existing file; return None if nothing to do."""
    existing = load_existing()
    print(f"Existing file: {len(existing)} rows")

    print(f"Downloading season {CURRENT_SEASON}...")
    fresh = fetch_season(CURRENT_SEASON)
    if fresh.empty:
        print(f"  {CURRENT_SEASON}: no games yet, file left unchanged")
        return None

    season_id = fresh['SEASON_ID'].iloc[0]
    kept = existing[existing['SEASON_ID'] != season_id]
    print(f"  {CURRENT_SEASON} (SEASON_ID {season_id}): "
          f"{len(existing) - len(kept)} rows replaced by {len(fresh)}")

    return pd.concat([kept, fresh], ignore_index=True)[existing.columns]


def main(incremental=False):
    games_df = fetch_incremental() if incremental else fetch_full()
    if games_df is None:
        return

    games_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(games_df)} rows to {OUTPUT_PATH}")
    print(games_df.groupby('SEASON_ID').size().to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--incremental', action='store_true',
                        help='refetch only the current season instead of every season')
    args = parser.parse_args()
    main(incremental=args.incremental)

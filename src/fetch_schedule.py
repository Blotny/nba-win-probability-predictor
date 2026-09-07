"""Fetch scheduled (not yet played) games into data/raw/schedule.csv.

  python src/fetch_schedule.py                 # next 7 days
  python src/fetch_schedule.py --days 3        # shorter horizon
  python src/fetch_schedule.py --season 2026-27 --all

LeagueGameFinder (fetch_data.py) only ever returns games that have been PLAYED,
which is why the daily job needs a second source. ScheduleLeagueV2 carries the
full published season schedule months ahead of time.

Rows are written in the same shape as data/raw/games.csv - two per game, home
and away - with every box-score column blank. clean_data derives is_future from
that missing WL, so nothing downstream needs to know where a row came from.
"""

import argparse
import datetime as dt
import os

import pandas as pd
from nba_api.stats.endpoints import scheduleleaguev2

from fetch_data import CURRENT_SEASON

OUTPUT_PATH = 'data/raw/schedule.csv'

# gameId prefixes: 001 preseason, 002 regular season, 004 playoffs,
# 005 play-in, 006 NBA Cup final. The model is trained on regular season only.
REGULAR_SEASON_PREFIX = '002'

# Horizon is deliberately longer than the useful prediction set, because
# stale_games does the filtering - not a day count.
#
# Measured on the 2026-27 schedule (4 windows across the season): the
# stale_games == 0 set saturates at 13-14 games and stops growing after day 2-4.
# That ceiling is arithmetic - each of the 30 teams can contribute at most one
# game before it has an unresolved game of its own ahead of it, so the set can
# never exceed 15. But saturation lands on day 3 in some windows and day 4 in
# others, and a post-All-Star gap could push a team's first game later still,
# so a short horizon silently drops exact-state games.
#
# 7 days clears saturation everywhere measured. The extra stale_games > 0 rows
# cost nothing: predict_upcoming re-scores daily and keeps the newest prediction
# per game, so every game ends up stored with the freshest state available, and
# meanwhile Power BI gets a "coming up" view for free.
DEFAULT_DAYS = 7

# data/raw/games.csv column order, so the two files stack cleanly
RAW_COLUMNS = ['SEASON_ID', 'TEAM_ID', 'TEAM_ABBREVIATION', 'TEAM_NAME',
               'GAME_ID', 'GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'FGM',
               'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
               'FT_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV',
               'PF', 'PLUS_MINUS']


def season_id(season):
    """'2026-27' -> '22026' (the SEASON_ID convention in games.csv)."""
    return '2' + season.split('-')[0]

def fetch_schedule(season):
    df = scheduleleaguev2.ScheduleLeagueV2(season=season,
                                           league_id='00').get_data_frames()[0]
    df = df[df['gameId'].str.startswith(REGULAR_SEASON_PREFIX)].copy()
    df['GAME_DATE'] = pd.to_datetime(df['gameDate'])

    # NBA Cup knockout games are regular-season games (they count in the
    # standings) but the schedule lists them with the participants blank until
    # the group stage ends. There is nothing to predict about a game whose
    # teams are unknown, and a blank tricode produces a NaN MATCHUP that breaks
    # clean_data.fix_corrupted_matchup. They reappear here, filled in, once the
    # bracket is set.
    known = (df['homeTeam_teamTricode'].notna() & (df['homeTeam_teamTricode'] != '') &
             df['awayTeam_teamTricode'].notna() & (df['awayTeam_teamTricode'] != ''))
    if (~known).any():
        print(f"  skipping {int((~known).sum())} games with teams not yet decided "
              f"(NBA Cup knockout)")
    df = df[known]

    return df.sort_values(['GAME_DATE', 'gameId']).reset_index(drop=True)


def to_raw_rows(df, season):
    """One schedule row -> two games.csv-shaped rows (home + away)."""
    sid = season_id(season)

    def side(is_home):
        us, them = ('homeTeam', 'awayTeam') if is_home else ('awayTeam', 'homeTeam')
        sep = ' vs. ' if is_home else ' @ '
        out = pd.DataFrame({
            'SEASON_ID': sid,
            'TEAM_ID': df[f'{us}_teamId'],
            'TEAM_ABBREVIATION': df[f'{us}_teamTricode'],
            'TEAM_NAME': df[f'{us}_teamCity'] + ' ' + df[f'{us}_teamName'],
            'GAME_ID': df['gameId'],
            'GAME_DATE': df['GAME_DATE'].dt.strftime('%Y-%m-%d'),
            'MATCHUP': df[f'{us}_teamTricode'] + sep + df[f'{them}_teamTricode'],
        })
        # every remaining column is box score: unknown until the game is played
        for col in RAW_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        return out[RAW_COLUMNS]

    return pd.concat([side(True), side(False)], ignore_index=True)


def main(season=None, days=DEFAULT_DAYS, fetch_all=False, today=None):
    season = season or CURRENT_SEASON
    print(f"Fetching {season} schedule...")
    sched = fetch_schedule(season)
    if sched.empty:
        print(f"  no regular-season games published for {season}")
        return

    print(f"  {len(sched)} regular-season games published "
          f"({sched['GAME_DATE'].min().date()} .. {sched['GAME_DATE'].max().date()})")

    if not fetch_all:
        today = pd.Timestamp(today or dt.date.today())
        window_end = today + pd.Timedelta(days=days - 1)
        sched = sched[(sched['GAME_DATE'] >= today) &
                      (sched['GAME_DATE'] <= window_end)]
        print(f"  horizon {today.date()} .. {window_end.date()}: {len(sched)} games")

    if sched.empty:
        print("  nothing in the horizon; writing an empty schedule file")

    rows = to_raw_rows(sched, season)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    rows.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(rows)} rows ({len(rows) // 2} games) to {OUTPUT_PATH}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', default=None, help='e.g. 2026-27')
    parser.add_argument('--days', type=int, default=DEFAULT_DAYS,
                        help=f'horizon in days from today (default {DEFAULT_DAYS})')
    parser.add_argument('--all', action='store_true', dest='fetch_all',
                        help='write the whole published season, ignoring --days')
    parser.add_argument('--today', default=None,
                        help='override today (YYYY-MM-DD). TESTING ONLY - a date '
                             'in the future skips the games between the last '
                             'played one and the window, and stale_games cannot '
                             'see games it was never given, so those rows claim '
                             'exact state on stale history. Check data_through '
                             'in upcoming_predictions.csv before trusting a row.')
    args = parser.parse_args()
    main(season=args.season, days=args.days, fetch_all=args.fetch_all,
         today=args.today)

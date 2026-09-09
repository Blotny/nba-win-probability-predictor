"""Write data/processed/teams.csv - a lookup table for the dashboard.

  python src/fetch_teams.py

predictions.csv and upcoming_predictions.csv identify teams by three-letter
abbreviation only. Power BI needs a one-row-per-team table to hang logos and
full names off, related to the fact tables on the abbreviation.

Logos come from ESPN because they are PNG. The NBA's own CDN serves SVG, which
Power BI's Image URL data category cannot render.
"""
import os

import pandas as pd
from nba_api.stats.static import teams

OUTPUT_PATH = 'data/processed/teams.csv'

LOGO_TEMPLATE = 'https://a.espncdn.com/i/teamlogos/nba/500/{slug}.png'

# ESPN's slug is the lowercased NBA abbreviation except for these two.
ESPN_SLUG_OVERRIDES = {'NOP': 'no', 'UTA': 'utah'}


def build():
    rows = []
    for t in teams.get_teams():
        abbr = t['abbreviation']
        slug = ESPN_SLUG_OVERRIDES.get(abbr, abbr.lower())
        rows.append({
            'team': abbr,
            'team_id': t['id'],
            'team_name': t['full_name'],
            'city': t['city'],
            'nickname': t['nickname'],
            'logo_url': LOGO_TEMPLATE.format(slug=slug),
        })
    return pd.DataFrame(rows).sort_values('team').reset_index(drop=True)


def main():
    df = build()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} teams to {OUTPUT_PATH}")
    print(df.head(3).to_string(index=False))


if __name__ == '__main__':
    main()

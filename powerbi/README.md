# Power BI dashboard - NBA win probability


Backtest dashboard for the saved model (`models/win_prob_blend.joblib`).

Only the **2025-26** season (`split = "test"`) is genuine out-of-sample - thevmodel was fit on 2020-21 … 2024-25. Filter visuals to `split = test` unless a page explicitly compares seasons.

  
## Prerequisites
1. Regenerate the data whenever the model or features change:

   ```

   python src/train_final_model.py    # refits + saves the model (rarely needed)

   python src/predict.py              # writes the 3 CSVs below

   ```

2. Power BI Desktop (Windows).

Data sources (all in `data/processed/`):

| file                   | grain                      | used for         |
| ---------------------- | -------------------------- | ---------------- |
| `predictions.csv`      | one row per game           | every page       |
| `calibration.csv`      | split × probability bucket | calibration page |
| `team_elo_history.csv` | one row per team per game  | team page        |
  
## 1. Load the data

**Home → Get data → Text/CSV** for each of the three files. Click **Transform

Data** (not just Load) and set column types, then **Close & Apply**.



`predictions`:

- `GAME_DATE` → Date

- `home_win_prob, away_win_prob, pred_margin, margin_win_prob, abs_error,`

  `logloss_contrib, brier_contrib, confidence, HOME_elo, AWAY_elo, elo_diff` → Decimal number

- `predicted_home_win, actual_home_win, correct, upset_flag, home_margin, SEASON_ID` → Whole number

- everything else → Text

  

`calibration`: `n` → Whole number; `mean_pred, mean_actual` → Decimal.

`team_elo_history`: `GAME_DATE` → Date; `elo_before` → Decimal; `is_home` → Whole number.

`prob_bucket` sorts correctly as text (`0-10%` … `90-100%`), no sort-by-column needed.


## 2. Measures

Create a measures table (**Home → Enter data**, name it `_Measures`, delete the

blank column) and add:


```DAX

Games              = COUNTROWS(predictions)

Accuracy           = AVERAGE(predictions[correct])

Log loss           = AVERAGE(predictions[logloss_contrib])

Brier              = AVERAGE(predictions[brier_contrib])

AUC not available  = BLANK()   -- AUC needs ranking; report from the notebook

Home win % actual  = AVERAGE(predictions[actual_home_win])

Home win % pred    = AVERAGE(predictions[home_win_prob])

Upsets             = SUM(predictions[upset_flag])

Baseline log loss  = 0.693

Accuracy vs baseline = [Accuracy] - [Home win % actual]

```

  
(For reference, the notebook reports test AUC ≈ 0.735.)

## 3. Pages

### Page 1 — Model performance  *(page filter: `split` = test)*

- 4 **Card** visuals: `Games`, `Accuracy`, `Log loss`, `Brier`.

- **Line chart**: axis `GAME_DATE` (month level), value `Log loss`. Add a

  constant line at 0.693 (Analytics pane) as the baseline.

- **Line chart**: axis `GAME_DATE` (month), value `Accuracy`.

- **Clustered column** *(remove the split filter on this one)*: axis

  `season_label`, values `Home win % actual` and `Home win % pred` — shows the

  model tracks home-court advantage per season.


### Page 2 — Calibration  *(use the `calibration` table; slicer `split`, default test)*

- **Line chart**: axis `prob_bucket`, values `mean_pred` and `mean_actual`.

  Overlapping lines = well calibrated.

- **Scatter chart**: X `mean_pred`, Y `mean_actual`, size `n`, details

  `prob_bucket`. Points on the diagonal = calibrated.

- **Table**: `prob_bucket`, `n`, `mean_pred`, `mean_actual`.


### Page 3 — Predictions explorer  *(slicers: `split` (default test), `season_label`)*

- **Table**: `GAME_DATE`, `HOME_TEAM_ABBR`, `AWAY_TEAM_ABBR`, `home_win_prob`,

  `predicted_home_win`, `actual_home_win`, `correct`, `home_margin`,

  `pred_margin`.

- Conditional formatting: colour scale on `home_win_prob`; icon on `correct`.

- Sort by `GAME_DATE`.


### Page 4 — Upsets & confidence  *(page filter: `split` = test)*

- **Table**, filtered `upset_flag = 1`, sorted `abs_error` desc: `GAME_DATE`,

  teams, `home_win_prob`, `actual_home_win`, `home_margin`.

- **Clustered column**: axis `prob_bucket`, value `Games`, legend `correct` —

  where the model is right vs wrong by confidence level.

- **Histogram** (column chart): axis `confidence` (binned, size 0.1), value `Games`.


### Page 5 — Team view  *(optional — needs the extra query in step 4)*

- **Slicer**: `Teams[team]`.

- **Line chart**: axis `team_elo_history[GAME_DATE]`, value

  `team_elo_history[elo_before]` — Elo trajectory.

- **Cards**: team actual win rate vs average predicted win prob (measures on

  `team_games`, see below).

- **Clustered column**: axis `season_label`, values predicted vs actual wins.


### Page 6 — Upcoming games  *(uses `upcoming_predictions.csv`)*

The other five pages are a backtest. This one is the live product, and after the
first retrain on a finished 2026-27 it is the **only** honest measurement left -
at that point no season is out-of-sample any more.

**Load** `data/processed/upcoming_predictions.csv` as a new query named
`upcoming`. Types: `GAME_DATE`, `data_through` → Date; `predicted_at` → Date/Time;
`home_win_prob, away_win_prob, pred_margin, margin_win_prob, HOME_elo,`
`AWAY_elo, elo_diff, confidence` → Decimal; `predicted_home_win, stale_games,`
`is_early_season` → Whole number; `role`, `model_version` → Text; rest → Text.

Do **not** relate it to `predictions` - the two never hold the same game at the
same time. A game sits in `upcoming` until it is played, then appears in
`predictions`.

**Two models, on purpose.** Pages 1-5 show **v1** (fit 2020-25, holdout 2025-26);
page 6 shows **v2** (fit 2020-26) by default. This is not a mistake and not
drift - the two pages answer different questions. A backtest needs a model with
a season it never saw, or its accuracy figures are meaningless. A forecast has
no such constraint: the season ahead is unseen by every candidate, so it should
use the model fit on the most, and the most recent, data. Filter this page to
`role = "primary"` and put `model_version` on a card so the reader always knows
whose number they are looking at. `role = "shadow"` holds v1 scoring the same
fixtures, for the season-long comparison.

**Calculated columns** on `upcoming`:

```DAX
Pick      = IF(upcoming[home_win_prob] >= 0.5,
               upcoming[HOME_TEAM_ABBR], upcoming[AWAY_TEAM_ABBR])

Pick prob = MAX(upcoming[home_win_prob], 1 - upcoming[home_win_prob])

Matchup   = upcoming[AWAY_TEAM_ABBR] & " @ " & upcoming[HOME_TEAM_ABBR]

State     = SWITCH(TRUE(),
                upcoming[stale_games] = 0,  "Exact",
                upcoming[stale_games] <= 3, "Stale (~2pp)",
                                            "Preview only")

State ord = SWITCH(TRUE(),
                upcoming[stale_games] = 0,  1,
                upcoming[stale_games] <= 3, 2,
                                            3)
```

Sort `State` by `State ord` (Column tools → Sort by column).

**Measures:**

```DAX
Upcoming games   = COUNTROWS(upcoming)
Exact-state      = CALCULATE(COUNTROWS(upcoming), upcoming[stale_games] = 0)
Last scored      = MAX(upcoming[predicted_at])
Model            = MAX(upcoming[model_version])
```

> The Power BI UI here is **English**. Menu names below match it.

### Page 6 layout - master / detail

Pick a game on the left, read it on the right. The interaction is native
cross-filtering; nothing is scripted.

```
+---------------------------+ +-------------------------------------------+
| SLICER  State             | | [logo]      CLE  vs  WAS      [logo]      |
+---------------------------+ |                                           |
| CARD Upcoming | CARD Exact| | CARD home team      CARD away team        |
+---------------------------+ | CARD 83%            CARD 17%              |
|                           | +-------------------------------------------+
|  TABLE                    | | STACKED BAR  home_win_prob | away_win_prob|
|  the games                | +-------------------------------------------+
|  (click a row)            | | CARD pred_margin | CARD elo_diff          |
|                           | | CARD State       | CARD data_through      |
+---------------------------+ +-------------------------------------------+
| CARD Model | CARD Last scored                                           |
+-------------------------------------------------------------------------+
```

**Page filter:** `role = "primary"` (v2). Put it on the page, not per visual.

**Left - the picker**

1. **Slicer**: `State`. Leave it unfiltered by default.
2. **Cards**: `Upcoming games`, `Exact-state`.
3. **Table** - the master visual. Fields: `GAME_DATE`, `Matchup`, `Pick`,
   `Pick prob`, `pred_margin`, `State`. Sort by `GAME_DATE`, then `stale_games`.
   Conditional formatting: colour scale on `Pick prob`, background colour on
   `State` (green / amber / grey).

**Right - the detail**, all of it driven by whichever row is selected:

4. **Two Card visuals** for the logos, fed from `upcoming[Home logo]` and
   `upcoming[Away logo]` - the calculated columns below, **not** the team table
   directly (see the trap). Use **Card**, not Table.

   Table/Matrix visuals in Power BI have no real show/hide toggle for the
   header row - **Format your visual → Column headers** is styling only
   (font, colour, background), not visibility. A single-column Table used for
   an image ends up with the field name floating above the crest and no way to
   remove it short of colour-matching the header text to its background. Card
   sidesteps the problem entirely: drop `upcoming[Home logo]` into a Card's
   field well and there is no header row to fight - only **Category label**
   under Format your visual, a normal on/off toggle, which also drives the
   label under every numeric card elsewhere on this page.

   Row selection still filters the image, same as it would from a Table -
   it's a column of `upcoming` either way, so the interaction mechanism is
   unchanged. Resize via **Format your visual → Callout value → Image
   size**.
5. **Cards**: `upcoming[Home name]`, `upcoming[Away name]`, `home_win_prob`,
   `away_win_prob` (format as percentage, 0 decimals).
6. **Stacked bar chart** - the probability bar. Y axis: empty, or `Matchup`.
   Values: `home_win_prob` **and** `away_win_prob`, both. They sum to 1, so the
   bar is full without the 100% variant, which would rescale for nothing.
   Fixed colours - gold for the favourite, navy for the other. The reference
   app does the same; it does not use club colours.
7. **Cards**: `pred_margin`, `elo_diff`, `State`, `data_through`.

**Bottom**: `Model` and `Last scored`, so the reader always knows which model
produced the numbers and how fresh they are.

### Logos - the two-relationship trap

`teams.csv` (from `src/fetch_teams.py`) is one row per team: `team`, `team_id`,
`team_name`, `city`, `nickname`, `logo_url`. Logos are ESPN PNGs; the NBA's own
CDN serves SVG, which the Image URL category cannot render.

A game row names **two** teams, so you need two relationships into the same
table - and Power BI keeps only one active. Do not fight that with
`USERELATIONSHIP`; load the query twice.

1. Get data → Text/CSV → `teams.csv`, name it `teams_home`
2. Right-click it in the Data pane → **Duplicate**, name the copy `teams_away`
3. Model view: `teams_home[team]` → `upcoming[HOME_TEAM_ABBR]` and
   `teams_away[team]` → `upcoming[AWAY_TEAM_ABBR]`. Both one-to-many, single
   direction, filtering *from* the team table *to* `upcoming`.

**Then do NOT point the logo visual at `teams_home[logo_url]`.** A single-
direction relationship filters teams → upcoming, so selecting a row in
`upcoming` does not propagate back: the visual keeps listing all 30 teams and
looks broken. Switching the relationship to "Both" fixes the symptom and
invites ambiguity elsewhere in the model. Pull the values onto the fact table
instead - `RELATED` runs from the many side to the one side, which is the
direction the model already has:

```DAX
Home logo = RELATED(teams_home[logo_url])
Away logo = RELATED(teams_away[logo_url])
Home name = RELATED(teams_home[team_name])
Away name = RELATED(teams_away[team_name])
```

4. Set **Column tools → Data category → Image URL** on `Home logo` and
   `Away logo`.

Because these are columns of `upcoming`, row selection filters them for free.
30 rows loaded twice costs nothing, and each visual reads the side it means.

### Conditional formatting on the table

Select the table → **Format your visual** → **Cell elements** → pick a field
from the dropdown.

- `Pick prob`: **Background color** on → **fx** → Format style **Color scale**
  → set Minimum/Maximum colours.
- `State`: **Background color** on → **fx** → Format style **Rules** → three
  rules: `State = "Exact"` → green, `State` contains `"Stale"` → amber,
  `State = "Preview only"` → grey.

### The blank-selection state ("2x Atlanta Hawks")

With nothing selected in the table, a Card bound to a **column** (`Home name`,
`Home logo`, ...) cannot resolve one row out of 52, so it silently falls back to
the alphabetically smallest value in that column - "Atlanta Hawks" for both
Home and Away, since both draw from the same 30-team pool. Not a bug, just
Card's default aggregation on an unfiltered text column.

**Fix for text/number cards - convert the column to a measure.**
`SELECTEDVALUE` returns blank both when nothing is selected and when several
rows are (ctrl-click), so one guard handles both:

```DAX
Home name (m) =
VAR sel = SELECTEDVALUE(upcoming[Home name])
RETURN IF(ISBLANK(sel), "Select a game ↓", sel)
```

Same pattern for `Away name`, `pred_margin`, `elo_diff`, `State`,
`data_through`. Swap the card fields to these measures.

**Logos can't use this trick.** Image URL data category only works on a
physical column, never on a measure - so `Home logo`/`Away logo` must stay
columns, and there is no placeholder-on-blank option for them.

**Practical default: select a game, then save.** Row selection is part of a
report's saved state - click a specific game (e.g. an opening-night one),
save the .pbix (or publish), and that selection is what shows on next open,
until someone clicks elsewhere. Combine both: measures give the text cards a
clean placeholder, and a saved selection gives the logos something real
instead of relying on the placeholder text alone.

### Edit interactions - the tutorial

By default Power BI **highlights** in charts and **filters** in tables and
cards. Highlight dims the non-matching part instead of removing it, so a detail
panel ends up showing a faded version of every other game behind the one you
picked. For master/detail you want Filter.

1. Click the **table** once to select it. Interactions are always configured
   *from* the visual that does the filtering.
2. Ribbon → **Format** tab → **Edit interactions**. The Format tab only appears
   while a visual is selected.
3. Every *other* visual now shows a small row of icons in its top-right corner:

   | icon | meaning |
   | ---- | ------- |
   | funnel | **Filter** - the target shows only the selected row |
   | chart with one faded bar | **Highlight** - dims instead of removing |
   | circle with a slash | **None** - the target ignores the selection |

4. Click the **funnel** on every right-hand visual: both logos, all the cards,
   the stacked bar.
5. Click the **circle-slash** on the two left cards (`Upcoming games`,
   `Exact-state`). Those are meant to count the whole slate - if they filter,
   both drop to 1 the moment you click a game, which is useless.
6. Click **Edit interactions** again to leave the mode.

Then repeat starting from the **slicer**: select it, Edit interactions, set the
table and the two left cards to Filter (they should narrow when a `State` is
picked), and the right-hand detail visuals to None, so changing the slicer does
not blank the detail panel.

**Behaviour worth knowing:** click a row to select, ctrl+click to add rows,
click empty canvas to clear. With nothing selected the detail cards show the
aggregate over all 52 games, which reads as nonsense. A default selected row is
not possible in Power BI, so either accept it or drop a text box on the panel
saying "select a game".

### Where Power BI will not follow

The card-per-game scrolling layout, rounded corners, an expanding "more"
section - Power BI lays out a grid of visuals, not free-form tiles. Get close
with table plus detail panel and stop there. If that exact look is the goal,
it is a web page: the Streamlit option reading these same CSVs is about a day's
work and gives a shareable link a `.pbix` never will.

**Why every game is shown, marked rather than hidden.** `stale_games > 0` does
not mean "no prediction is possible" - the model always returns a probability.
It means the inputs are N games out of date. `tests/test_future_features.py`
measures what that actually costs:

| stale_games | games | mean shift | max shift | pick flips |
| ----------- | ----- | ---------- | --------- | ---------- |
| 0           | 15    | 0.0000     | 0.0000    | 0          |
| 1           | 13    | 0.0206     | 0.0462    | 0          |
| 2           | 15    | 0.0197     | 0.0531    | 1          |
| 3           | 15    | 0.0170     | 0.0417    | 0          |

Drift does not grow past level 1 - it plateaus around 2pp, and the predicted
winner changed in 1 of 43 stale games (2.3%). Against a model that mostly
outputs 0.35-0.75, 2pp is noise. Stale rows are worth reading; they are just
not worth *scoring*.

**`data_through` is the second half of the check.** `stale_games` counts
unresolved games *inside the fetched window* - it cannot know about games that
exist in the schedule but were never fetched. Every row therefore records
`data_through`, the last played game its features rest on. A `stale_games = 0`
row is only genuinely exact if no games were played between `data_through` and
`GAME_DATE`. Out of season that holds trivially (nothing is played between
April and October); mid-season it holds because the daily window starts today.
It breaks only if `fetch_schedule.py --today` is pointed at a future date, which
is a testing flag - see its help text.

**So: filter live metrics to `stale_games = 0`, but not the table.** Any card
or chart reporting live accuracy or log loss must sit behind
`upcoming[stale_games] = 0`, or the track record mixes two qualities of
prediction into one number.

### Seeing anything before the season starts

`fetch_schedule.py` defaults to a 7-day horizon, so out of season the file is
empty and this page is blank. For a preview, widen it once:

```
python src/fetch_schedule.py --days 50
python src/clean_data.py
python src/feature_engineering.py
python src/predict_upcoming.py
```

These opening-night predictions are **not** provisional: no games are played
between now and 20 October, so Elo cannot move, and a `stale_games = 0` row
scored today is identical to the one that would be scored on match day.

Once the season is running, drop back to the default - `run_daily.bat` already
calls `fetch_schedule.py` without `--days`.

## 4. Optional — team-level table for Page 5


`predictions` has a home team and an away team per row, so a single team slicer

needs an unpivoted table. In Power Query:
  

1. Right-click `predictions` → **Reference**, rename `team_games_home`.

   Keep `GAME_ID, GAME_DATE, season_label, split, HOME_TEAM_ABBR,`

   `home_win_prob, actual_home_win, home_margin`.

   Add columns: `team = [HOME_TEAM_ABBR]`, `is_home = 1`,

   `team_win_prob = [home_win_prob]`, `team_won = [actual_home_win]`,

   `team_margin = [home_margin]`. Remove the original 4 columns.

2. Same again as `team_games_away` from the AWAY side:

   `team = [AWAY_TEAM_ABBR]`, `is_home = 0`,

   `team_win_prob = 1 - [home_win_prob]`, `team_won = 1 - [actual_home_win]`,

   `team_margin = -[home_margin]`.

3. **Append Queries** → new query `team_games` = home + away.

4. `Teams` query: `team_games` → keep `team` → **Remove Duplicates**.

5. **Model view**: relate `Teams[team]` → `team_games[team]` and

   `Teams[team]` → `team_elo_history[team]` (both one-to-many, single direction).

   Now one `Teams[team]` slicer filters the Elo chart and the game table together.

  

Measures on `team_games`:

```DAX

Team games        = COUNTROWS(team_games)

Team win rate     = AVERAGE(team_games[team_won])

Team pred win rate = AVERAGE(team_games[team_win_prob])

```

  

## 5. Refresh workflow

1. `python src/predict.py` (and `train_final_model.py` first if the model changed).

2. Power BI Desktop → **Home → Refresh**.

  
If the repo moves, fix the file paths under **Transform data → Data source settings**.


## Notes

- The first ~10 games of 2020-21 all get ≈0.52 — every feature is still NaN / 1500

  Elo at season start, so the model can't separate teams yet. Expected, not a bug.

- `train` / `val` rows in `predictions.csv` are in-sample (the model saw them).

  Keep pages filtered to `split = test` for honest numbers.

- Both prediction files now carry `model_version`, but they behave differently:
  `predictions.csv` is regenerated wholesale on every `predict.py` run and holds
  **one** version, while `upcoming_predictions.csv` is append-only and can hold
  several. After a retrain, pages 1-5 silently switch to the new model in full;
  page 6 legitimately shows both, which is what the `model_version` slicer is
  for.
- `predict_upcoming.py` backfills `actual_home_win`, `home_margin`, `correct`,
  `logloss_contrib` and `brier_contrib` for predictions whose game has since
  been played, so the live track record is computable in Power BI. Filter it to
  `stale_games = 0` and `actual_home_win` not blank.
- `.pbix` is a large binary — consider adding `powerbi/*.pbix` to `.gitignore`

  and keeping only this README under version control.

## Related notes

- [[NBA predictor]] — project map
- [[src pipeline]] — `predict.py` / `train_final_model.py` regenerate the data
- [[data and models]] — schemas for `predictions.csv`, `calibration.csv`, `team_elo_history.csv`
- [[model training notebook and conclusions]] — source for the AUC figure and metric definitions
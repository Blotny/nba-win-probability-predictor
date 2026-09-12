# Power BI dashboard — NBA win probability

Backtest dashboard for the saved model (`models/win_prob_blend.joblib`).
Only the **2025-26** season (`split = "test"`) is genuine out-of-sample — the
model was fit on 2020-21 … 2024-25. Filter visuals to `split = test` unless a
page explicitly compares seasons.

## Prerequisites

1. Regenerate the data whenever the model or features change:
   ```
   python src/train_final_model.py    # refits + saves the model (rarely needed)
   python src/predict.py              # writes the 3 CSVs below
   ```
2. Power BI Desktop (Windows).

Data sources (all in `data/processed/`):

| file | grain | used for |
|------|-------|----------|
| `predictions.csv` | one row per game | every page |
| `calibration.csv` | split × probability bucket | calibration page |
| `team_elo_history.csv` | one row per team per game | team page |

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
Games              = COUNTROWS(predictions)
Accuracy           = AVERAGE(predictions[correct])
Log loss           = AVERAGE(predictions[logloss_contrib])
Brier              = AVERAGE(predictions[brier_contrib])
AUC not available  = BLANK()   -- AUC needs ranking; report from the notebook
Home win % actual  = AVERAGE(predictions[actual_home_win])
Home win % pred    = AVERAGE(predictions[home_win_prob])
Upsets             = SUM(predictions[upset_flag])
Baseline log loss  = 0.693
Accuracy vs baseline = [Accuracy] - [Home win % actual]
```

(For reference, the notebook reports test AUC ≈ 0.735.)

## 3. Pages

### Page 1 — Model performance  *(page filter: `split` = test)*
- 4 **Card** visuals: `Games`, `Accuracy`, `Log loss`, `Brier`.
- **Line chart**: axis `GAME_DATE` (month level), value `Log loss`. Add a
  constant line at 0.693 (Analytics pane) as the baseline.
- **Line chart**: axis `GAME_DATE` (month), value `Accuracy`.
- **Clustered column** *(remove the split filter on this one)*: axis
  `season_label`, values `Home win % actual` and `Home win % pred` — shows the
  model tracks home-court advantage per season.

### Page 2 — Calibration  *(use the `calibration` table; slicer `split`, default test)*
- **Line chart**: axis `prob_bucket`, values `mean_pred` and `mean_actual`.
  Overlapping lines = well calibrated.
- **Scatter chart**: X `mean_pred`, Y `mean_actual`, size `n`, details
  `prob_bucket`. Points on the diagonal = calibrated.
- **Table**: `prob_bucket`, `n`, `mean_pred`, `mean_actual`.

### Page 3 — Predictions explorer  *(slicers: `split` (default test), `season_label`)*
- **Table**: `GAME_DATE`, `HOME_TEAM_ABBR`, `AWAY_TEAM_ABBR`, `home_win_prob`,
  `predicted_home_win`, `actual_home_win`, `correct`, `home_margin`,
  `pred_margin`.
- Conditional formatting: colour scale on `home_win_prob`; icon on `correct`.
- Sort by `GAME_DATE`.

### Page 4 — Upsets & confidence  *(page filter: `split` = test)*
- **Table**, filtered `upset_flag = 1`, sorted `abs_error` desc: `GAME_DATE`,
  teams, `home_win_prob`, `actual_home_win`, `home_margin`.
- **Clustered column**: axis `prob_bucket`, value `Games`, legend `correct` —
  where the model is right vs wrong by confidence level.
- **Histogram** (column chart): axis `confidence` (binned, size 0.1), value `Games`.

### Page 5 — Team view  *(optional — needs the extra query in step 4)*
- **Slicer**: `Teams[team]`.
- **Line chart**: axis `team_elo_history[GAME_DATE]`, value
  `team_elo_history[elo_before]` — Elo trajectory.
- **Cards**: team actual win rate vs average predicted win prob (measures on
  `team_games`, see below).
- **Clustered column**: axis `season_label`, values predicted vs actual wins.

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
Team games        = COUNTROWS(team_games)
Team win rate     = AVERAGE(team_games[team_won])
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
- `.pbix` is a large binary — consider adding `powerbi/*.pbix` to `.gitignore`
  and keeping only this README under version control.

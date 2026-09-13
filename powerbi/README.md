# Power BI dashboard - NBA win probability

Dashboard on top of [`nba-win-probability-predictor`](../README.md). 7 pages, built from the CSVs in `data/processed/` (regenerate via `run_daily.bat` or
the pipeline steps in the root README).

Pages 1–5 back-test the production model (**v1**, fit 2020-21…2024-25, with 2025-26 held out — only `split = "test"` rows are genuine out-of-sample). Page 7 is the live product: predictions on games not yet played, scored by **v2** (fit on every season including the 2025-26 holdout — a forecast has no holdout to protect, so it uses the model with the most, freshest data).

## Pages

**1 - Model performance** *(filter: `split = test`)*
Headline backtest numbers: games, accuracy, log loss, Brier score, and which model version produced them. Log loss and accuracy over time, plus predicted vs. actual home-win rate by season (shows the model tracking real home-court advantage, not just the baseline).

**2 - Calibration**
Predicted vs. actual win rate by probability bucket, from `calibration.csv`. Points on the diagonal mean the model's stated probabilities are trustworthy - a 70% pick should win about 70% of the time. Only `train`/`test` splits exist here; the production model fits on train+val, so there's no genuine `val` holdout to show.

**3 - Predictions explorer**
Every played game, filterable and sortable: teams, predicted probability, actual result, margin. Slicers for split and season — for browsing individual games rather than reading aggregate metrics.

**4 - Upsets & confidence** *(filter: `split = test`)*
Games the model got most wrong, ranked by how big the surprise was. Also breaks down accuracy by confidence level — whether the model is actually more reliable when it claims to be.

**5 - Team view**
One team at a time: its Elo trajectory over the full history, and predicted vs. actual win totals per season. Picked via a team slicer.

**6 - All teams Elo**
Elo rating for every team over time, from `team_elo_history.csv`.

**7 - Upcoming games**
Live predictions for fixtures that haven't been played yet — team logos, predicted winner and margin, and a per-fixture staleness flag (`stale_games`) showing how current the underlying form/Elo data is. Once a game is settled, its actual result is backfilled and a live accuracy/log loss track record builds up, filtered to `stale_games = 0` for an honest number. Shows model **v2** (`role = primary`) by default, with **v1** riding along as a shadow (`role = shadow`) for comparison over the season.

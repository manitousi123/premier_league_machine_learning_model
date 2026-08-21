# Premier League Predictor — Build Plan

**Written:** 2026-08-21
**Status of base model:** working, committed, `precision = 0.625` on the 2022 test split.

---

## 1. Where we actually are today

| Thing | Current state |
|---|---|
| Data | `data/matches.csv`, 1,389 rows, 2020-09-12 to 2022-04-25 (2 partial seasons) |
| Model | `RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)` |
| Target | `result == "W"` — "did **this team** win?" (binary) |
| Features | `venue_code`, `opp_code`, `hour`, `day_code` + 3-match rolling means of `gf, ga, sh, sot, dist, fk, pk, pkatt` |
| Validation | one fixed split: train `< 2022-01-01`, test `> 2022-01-01` |
| Score | precision 0.625 on the "win" class |
| Where it lives | entirely inside `notebooks/main.ipynb` |
| Stack | Python + pandas 3.0.5, numpy 2.4.6, scikit-learn 1.9.0, jupyter |

This is a solid learning model. It is **not yet** something that can predict next weekend's fixtures. The gap is mostly plumbing and correctness, not machine learning.

---

## 2. Five things that must be fixed before any UI gets built

These are not nice-to-haves. Each one will silently produce wrong predictions on future seasons if left alone.

### 2.1 Team names don't match between columns — BLOCKER

The `team` column and the `opponent` column use different names for the same club:

```
team column               opponent column
------------------------  ---------------
Manchester United         Manchester Utd
Wolverhampton Wanderers   Wolves
Brighton and Hove Albion  Brighton
Newcastle United          Newcastle Utd
Tottenham Hotspur         Tottenham
West Ham United           West Ham
Sheffield United          Sheffield Utd
West Bromwich Albion      West Brom
```

Every match sits in the table **twice** — once from each team's point of view. To turn two team-rows into one match prediction ("Arsenal 62% to beat Chelsea") you must join those two rows together. With mismatched names that join silently drops those 8 clubs.

**Fix:** one canonical name map in `src/plfootball/teams.py`, applied to both columns at load time. Add a hard check that fails loudly if an unmapped name appears.

### 2.2 Liverpool is missing from the 2022 season — data gap

Liverpool appears 38 times as a `team` (2021 only) and 33 times as an `opponent` in 2022, but has **zero** `team` rows in 2022. The 2022 season has 19 teams in this file, not 20. The scrape that produced this CSV dropped a club.

**Fix:** automated data-quality checks that run on every ingest (see Phase 4). A missing club is the kind of thing that quietly ruins a model.

### 2.3 Category codes are unstable across seasons — BLOCKER

```python
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
```

`.cat.codes` assigns numbers alphabetically **from whatever teams happen to be in the dataframe**. Promotion and relegation change that list every single season. `opp_code = 7` might mean Everton this year and Fulham next year. The model learns a mapping that quietly becomes nonsense.

**Fix:** a frozen, explicit mapping saved alongside the model, covering every club that has ever appeared, with a defined slot for unseen/newly-promoted clubs.

### 2.4 The split-date validation can't tell you if you're improving

One hard-coded date (`2022-01-01`) gives you one number from one slice of one season. Change a hyperparameter, the number moves, and you can't tell signal from noise. This directly blocks your third goal.

**Fix:** walk-forward backtest + a frozen benchmark set (see section 5).

### 2.5 Precision alone is the wrong scoreboard for a UI

`precision = 0.625` means: of the matches we called a win, 62.5% were wins. It says nothing about the ones we passed on, and nothing about **confidence**. A UI showing "Arsenal 91%" vs "Arsenal 51%" needs calibrated probabilities — `predict_proba` numbers that actually mean what they say.

**Fix:** switch the headline metrics to log loss and Brier score, add a calibration curve, keep precision/accuracy as secondary.

---

## 3. Target project layout

```
premier_league_machine_learning_project/
├── data/
│   ├── raw/                  # exactly as downloaded, never hand-edited
│   ├── processed/            # cleaned + feature-engineered parquet
│   └── benchmark/            # FROZEN eval set, committed to git
├── src/plfootball/
│   ├── teams.py              # canonical names + frozen team→code map
│   ├── ingest.py             # fetch results & upcoming fixtures
│   ├── quality.py            # data-quality assertions
│   ├── features.py           # rolling form, all pre-match only
│   ├── model.py              # train / load / predict
│   ├── evaluate.py           # metrics in one place
│   ├── backtest.py           # walk-forward evaluation
│   ├── registry.py           # model versions + their scores
│   └── predict.py            # upcoming gameweek → predictions
├── experiments/runs/         # one folder per experiment, metrics.json
├── models/                   # versioned model files + model cards
├── app/streamlit_app.py      # the UI
├── scripts/                  # refresh-data, train, backtest, predict
├── tests/
├── notebooks/main.ipynb      # exploration ONLY, never the source of truth
└── plans.md
```

**Rule to hold onto:** the notebook is a scratchpad. Anything the UI depends on lives in `src/`, is importable, and is tested. Notebooks can't be tested, diffed, or trusted.

---

## 4. Phase-by-phase plan

Each phase ends in something that works. Do them in order.

### Phase 0 — Repo groundwork (~half a day)

- [ ] Create the folder layout above
- [ ] `pip install -e .` so `import plfootball` works everywhere
- [ ] Move `matches.csv` to `data/raw/`, leave it untouched from now on
- [ ] Add `pytest` and `ruff` to requirements
- [ ] Extend `.gitignore`: `data/raw/*` (large and refetchable), `models/*.pkl`, `experiments/runs/*/artifacts/`
- [ ] **Keep committed:** `data/benchmark/`, all `metrics.json`, model cards
- [ ] Add a real `README.md` — the current one is 84 bytes of mojibake

**Done when:** `pytest` runs, and `python -c "import plfootball"` works.

### Phase 1 — Correctness fixes (~1 day)

- [ ] `teams.py`: canonical name map, applied to `team` and `opponent`
- [ ] `teams.py`: `TEAM_CODES` frozen dict + `UNKNOWN_TEAM = -1` for new promotions
- [ ] Replace both `.cat.codes` lines with lookups against that dict
- [ ] Port `rolling_averages` into `features.py`, with the pandas 3.0 fix already applied (`reset_index(level="team")`, not `droplevel`)
- [ ] `features.py` documents a **feature contract**: no feature may use any information from the match being predicted. `gf`, `ga`, `sh`, `xg` and friends are *outcomes* — they may only ever be used as rolling history, never as raw predictors
- [ ] Tests: name map is complete; codes stay stable when a team is removed from the input; no feature leaks the current match

**Done when:** rebuilt features reproduce the current model's score, and those tests pass.

### Phase 2 — Match-level predictions (~1 day)

Right now the model outputs one row per team, which can contradict itself — both sides of a match can be predicted to win.

- [ ] `features.py`: join the two team-rows into one match row (home form and away form side by side)
- [ ] Decide the target — **recommended: 3-class Home / Draw / Away** rather than binary win/not-win. Draws are about a quarter of matches, and a betting UI that can't say "draw" is half-blind
- [ ] Output `predict_proba` — three numbers summing to 1
- [ ] Test: every match appears exactly once, probabilities sum to 1, no match has two winners

**Done when:** one row per match, with three calibrated probabilities.

### Phase 3 — The evaluation harness (~2 days) — **your third goal lives here**

This is the piece that makes "did my change help?" answerable. Build it before you start tuning anything.

- [ ] **`data/benchmark/`** — a frozen holdout set, carved out once and never touched again. Recommended: the most recent full season. Committed to git so the comparison is identical forever. This is your "set piece of data."
- [ ] **`backtest.py`** — walk-forward evaluation. Train on everything before gameweek N, predict gameweek N, step forward, repeat across the whole benchmark. This mirrors how the model will really be used, and gives you dozens of scores instead of one.
- [ ] **`evaluate.py`** — one function, one metrics dict:
  - log loss (primary — punishes confident wrongness)
  - Brier score
  - accuracy, precision/recall per class
  - calibration curve data
  - ROI on flat stakes, once odds are in the data
- [ ] **Baselines to beat.** A number means nothing on its own:
  1. always predict home win (~45% accuracy in the PL)
  2. home/draw/away base rates
  3. bookmaker implied probability — the honest bar
  4. **v0** = today's model, log loss recorded as the starting line
- [ ] **`experiments/runs/<run_id>/metrics.json`** — every training run writes one, containing: git commit, data snapshot date, feature list, hyperparameters, all metrics, wall-clock time
- [ ] **`scripts/leaderboard.py`** — reads every `metrics.json` and prints one table, best first. This is the single command that answers "is it better?"

**Done when:** `python scripts/leaderboard.py` shows v0 alongside the baselines, and re-running the backtest twice gives identical numbers.

**Non-negotiable rule:** the benchmark set is never used for training, tuning, or feature selection. The moment you peek at it to make a decision, it stops measuring anything. Do hyperparameter work on a separate validation split.

### Phase 4 — Live data pipeline (~2 days)

Your data stops in April 2022. Nothing can predict "upcoming" fixtures without a live feed.

- [ ] Pick a source (see section 7 — **recommended: football-data.co.uk**)
- [ ] `ingest.py`: `fetch_results(season)` writes to `data/raw/`
- [ ] `ingest.py`: `fetch_fixtures()` returns upcoming matches with date, kickoff time, home, away
- [ ] `quality.py` assertions, run on every ingest:
  - every club has the expected number of matches
  - exactly 20 clubs per PL season
  - no unmapped team names
  - no duplicate match rows
  - no impossible values (negative shots, 30 goals)
  - **the Liverpool-2022 bug would have been caught by the first check**
- [ ] Backfill more history — more seasons is the single cheapest accuracy win available to you
- [ ] `scripts/refresh_data.py`: one command, fetch then validate then rebuild features

**Done when:** one command pulls current results and next week's fixtures, and loudly refuses to write bad data.

### Phase 5 — Gameweek predictions (~1 day)

- [ ] `predict.py`: `predict_gameweek(date_from, date_to)` returns one row per fixture with three probabilities
- [ ] Handle the cold-start case: a newly promoted club has no 3-match rolling form. Decide and document the fallback — league-average form is the simple answer
- [ ] Write predictions to `data/processed/predictions_<gameweek>.parquet` **before** the matches are played, so you accumulate a genuine real-world track record. That is worth more than any backtest.
- [ ] `scripts/score_last_gameweek.py`: fill in actual results, append to a running accuracy log

**Done when:** you can generate next weekend's predictions with one command.

### Phase 6 — The UI (~3 days)

**Recommended stack: Streamlit.** Pure Python, no JavaScript, and it turns this into a real app in a few hundred lines. FastAPI + React is the "proper" answer but it's roughly 5x the work for a tool that may only ever have one user. Start with Streamlit; you can always put a React front end on it later.

Four pages:

**1. This Gameweek** — the main view
- Card per fixture: badges, kickoff time, three probability bars
- The model's pick, plus a confidence label (high / medium / low)
- Recent form for both sides (last 5 results)
- Where odds exist: implied probability side by side with ours, and any gap flagged

**2. Track Record**
- Accuracy and log loss over time, as a line chart
- Predicted vs actual, gameweek by gameweek
- Calibration plot — when we say 70%, does it happen 70% of the time?
- Filters by club and by confidence band

**3. Model Comparison** — your third goal, made visible
- The leaderboard table, straight from `experiments/runs/`
- Pick any two runs, see their metrics diffed side by side
- Which fixtures did they disagree on, and who was right
- Feature importances

**4. Team Explorer**
- Per-club form trends, home vs away splits, upcoming difficulty

Notes:
- The UI **reads** saved predictions and metrics. It never trains a model. Keep it a thin, fast reader over parquet and JSON files.
- Cache with `@st.cache_data`.
- Every probability shown gets an "as of" timestamp.

**Done when:** `streamlit run app/streamlit_app.py` shows real predictions for the next gameweek.

### Phase 7 — Automation and the improvement loop (~2 days)

- [ ] Weekly scheduled job: refresh data, score last gameweek, predict next
- [ ] GitHub Actions on push: run tests and the backtest, comment the leaderboard diff on the PR — so you can never merge an accuracy regression by accident
- [ ] `models/<version>/MODEL_CARD.md`: what changed, what it scored, what it's bad at

**Then, and only then, start improving the model.** In rough order of expected payoff:

1. **More history** — 10 seasons instead of 2. Biggest single win, least effort.
2. **Better features** — Elo ratings, days since last match, rolling xG (much better signal than rolling goals), home/away splits, league position, goal difference
3. **Rolling window length** — 3 matches is arbitrary; backtest 3 vs 5 vs 10
4. **Model class** — gradient boosting (XGBoost / LightGBM) usually beats random forest on tabular data like this
5. **Probability calibration** — `CalibratedClassifierCV`; forests are known to be poorly calibrated out of the box
6. **Hyperparameter search** — last, and lowest payoff. `n_estimators=50` is small, but tuning is worth maybe a point or two.

Every one of these is a single experiment: change one thing, run the backtest, check the leaderboard, keep or discard. That's the whole loop.

---

## 5. How "did it get better?" works in practice

```bash
# 1. try something
git checkout -b exp/rolling-xg

# 2. train and evaluate — writes experiments/runs/<id>/metrics.json
python scripts/train.py --config configs/rolling_xg.yaml

# 3. compare against everything you've ever tried
python scripts/leaderboard.py
```

```
run_id              log_loss  brier  acc    roi     data_thru   note
------------------- --------- ------ ------ ------- ----------- ----------------------
exp_rolling_xg       0.981    0.198  0.541  +2.1%   2026-08-15  rolling xG features
v0_baseline_rf       1.043    0.213  0.512  -1.4%   2022-04-25  original notebook model
bookmaker_implied    0.968    0.194  0.549   0.0%   2026-08-15  the bar to beat
always_home          1.312    0.281  0.451  -4.8%   2026-08-15  naive baseline
```

(Illustrative numbers — the real ones come out of Phase 3.)

Lower log loss is better. If your run doesn't beat `v0_baseline_rf`, you throw the change away — no arguing with the table. If it doesn't beat `bookmaker_implied`, you have an interesting model but not a profitable one.

**Keep the whole leaderboard in git.** Six months from now, the record of what *didn't* work is as valuable as what did.

---

## 6. Suggested order of work

| Order | Phase | Effort | Why here |
|---|---|---|---|
| 1 | Phase 0 — groundwork | 0.5d | everything else needs it |
| 2 | Phase 1 — correctness | 1d | fix the silent bugs before building on them |
| 3 | Phase 3 — eval harness | 2d | **before** tuning, so you can measure |
| 4 | Phase 4 — live data | 2d | no future predictions without it |
| 5 | Phase 2 — match-level | 1d | needs live data to be worth doing |
| 6 | Phase 5 — gameweek preds | 1d | first real output |
| 7 | Phase 6 — UI | 3d | now there's something to show |
| 8 | Phase 7 — automation | 2d | make it run itself |

**Roughly 12 to 13 focused days.** Phases 0 to 3 are the ones people skip and regret; they're the reason the rest goes quickly.

Note the eval harness comes *before* the model improvements. Tuning without a scoreboard is guessing.

---

## 7. Decisions I need from you

**1. Data source.** My recommendation is **football-data.co.uk** — free per-season CSVs going back to the 1990s, stable format, and it includes **bookmaker odds**, which gives you the honest baseline and ROI for free. Downside: no xG. FBref (where the current CSV came from) has xG and shot detail but requires scraping, breaks often, and is rate-limited. Best of both: football-data.co.uk as the backbone, FBref later for xG if you want it.

**2. Binary or 3-class.** I recommend switching to Home / Draw / Away. Draws are a quarter of all matches, and "not a win" is too vague for a UI.

**3. Streamlit or a real web app.** I recommend Streamlit for v1. Pure Python, days not weeks.

**4. Is betting a goal?** If yes, ROI and calibration become the primary metrics, and beating the bookmaker's implied probability becomes the actual target — a much higher bar than beating 0.625 precision. If it's just for interest, accuracy and a nice UI are enough. This changes what "better" means, so it's worth deciding early.

---

## 8. Things to watch out for

- **Don't let the notebook become the app.** Notebooks hide state, can't be tested, and diff terribly. `src/` is the product.
- **Never train on the future.** Any accuracy above roughly 60% on match outcomes almost always means leakage. Bookmakers with vast resources hover around 55%. If you beat them by a lot, you have a bug, not an edge.
- **The benchmark set is sacred.** Look at it only to report a final number.
- **pandas 3.0 changed `groupby.apply`** — grouping columns are no longer passed into your function. This already bit us once. Pin your versions.
- **Small data.** 1,389 rows is not much. Differences of 1 to 2% between models are probably noise. Trust the walk-forward average, not any single split.

---

## 9. First three things to do

1. `scripts/leaderboard.py` and the frozen benchmark set — you cannot improve what you cannot measure
2. `teams.py` with the canonical name map — unblocks match-level predictions
3. Pick a data source and backfill 10 seasons — the cheapest accuracy gain on this list

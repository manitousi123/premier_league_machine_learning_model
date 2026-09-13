# Premier League Predictor

A model that looks at an upcoming Premier League fixture and says whether a team
will win it — yes or no — along with how confident it is.

It works from two things: a rating for every club, built by replaying every match
since 2010, and how many chances each side has been creating lately. Those feed a
model trained on sixteen seasons of results, which returns a probability for each
team and one call.

Tested against thirteen seasons it never saw, it beats the ratings it is built
from and covers 90% of the distance from knowing nothing to matching a
bookmaker. It took cutting the model down from forty-two columns to three to get
there — see [What the backtest found](#what-the-backtest-found).

## Running it

```bash
pip install -e ".[dev]"
python scripts/build_dataset.py       # download, check, build the ratings and the table
python scripts/train_model.py         # grade the model on seasons it never saw
python scripts/predict_matchweek.py   # call every fixture in the coming week
```

The first downloads the results, checks them, and builds the ratings and the
training table. The second grades the model against three baselines on seasons
it was never trained on. The third is the one you actually use.

`notebooks/plea_ratings.ipynb` and `notebooks/feature_table.ipynb`
have the tables and charts; `docs/model-workflow.html` explains how it fits together.

## Predicting the next matchweek

```
10 FIXTURES - results current to Sun 06 Sep 2026

  Sat 12 Sep  15:00            Chelsea  63%  v  16%  Hull City         HOME will win
  Sat 12 Sep  20:00         Sunderland  18%  v  59%  Arsenal           AWAY will win
  Sun 13 Sep  16:30  Manchester United  36%  v  35%  Manchester City   HOME will not win
```

Every fixture is scored twice, once from each club's side, and each answer is a
yes or a no — so the pair lands in one of four places, not three:

```
                     AWAY WILL NOT WIN                      AWAY WILL WIN
                    +--------------------------------------+------------------------------+
HOME WILL WIN       |HOME team WILL win  (3)               |NO CALL - both backed  (0)    |
                    |  Chelsea      v Hull City    63%/16% |  none, as it should be       |
                    |  Liverpool    v Fulham       61%/16% |                              |
                    +--------------------------------------+------------------------------+
HOME WILL NOT WIN   |HOME team WILL NOT win  (5)           |AWAY team WILL win  (2)       |
                    |  Tottenham Ho v Everton      33%/38% |  Sunderland   v Arsenal 18%/59%
                    |  Manchester U v Manchester C 36%/35% |  Coventry Cit v Brighton 19%/57%
                    +--------------------------------------+------------------------------+
```

The top-right cell is the one to watch: both clubs backed to win the same
match is the model contradicting itself, so it is kept as its own outcome
rather than folded into a home win. It has never fired — 0 of 4,940 test
fixtures — but a contradiction reported as a confident call would be the worst
of the four.

The bottom-left is not a draw prediction. Draws are 27% there against 23%
overall, which is barely above chance. It means neither side is backed, and
that is roughly half of all matches — the model declining rather than failing.

Confidence means what it says: across the thirteen test seasons a call at 60%
landed 69% of the time and one at 70% landed 77%.

Re-run `build_dataset.py` after each matchweek so the ratings and form are
current, then `predict_matchweek.py`.

### The part that could break silently

Everything else in this project scores matches that already happened, where the
row was built from a completed result. A fixture on Saturday has no such row,
so one has to be constructed — and if it is assembled even slightly differently
from a training row, the model is shown numbers that do not mean what it
learned they meant. Nothing errors. The predictions are just quietly worse.

So the two quantities that matter are not recomputed in the predictor.
`form_sot` and `form_shots` come from `features.recent_form`, which shares the
long-form conversion and the window definition with the training path, and the
ratings come from `plea.ratings_entering`, which mirrors PLEA's own season
rollover.

The test that guards this replays real matchweeks as though they had not
happened — hiding everything from that day onward, rebuilding the rows, and
demanding they match what the training table holds for the same matches. Not
close: identical. Shifting the form window by one match, or the home bonus by
seventeen points, makes it fail.

## The prediction log

`data/predictions/log.csv` is the only file this project writes that cannot be
rebuilt. Everything else under `data/` regenerates from the raw results; a
record of what was said *before* a match was played does not, which is why it
is the one data file kept in git.

It exists because the backtest is a claim, not a verdict. The walk-forward says
the model should get 62% of calls right and score 0.5821 — whether it actually
does, week after week, on fixtures nobody had seen when the code was written,
is a different question. Only this file can answer it.

`predict_matchweek.py` maintains it: settle last week's calls against the
results that have since come in, then record this week's. The track record
splits by confidence band, because the whole claim being tested is that the
model's confidence means something:

```
          band  predictions  correct  hit_rate
neither backed           17        9       53%
        50-55%            3        0        0%
        55-60%            5        5      100%
        60-70%            4        3       75%
          70%+            2        1       50%
```

### Two ways to keep it honest

**A prediction cannot appear after the match.** The fixture list runs a few
days ahead of the results file, so a Sunday run will happily offer a call on
Saturday's matches — the model has not seen those results, nothing leaks, and
the call is genuinely blind. It is still not a forecast, and a record that
counts it as one is measuring something easier than the thing it claims to
measure. Those fixtures are dropped, and the run says how many. Same-day is
allowed: kickoffs run to the evening and the log works in whole days.

**Settling cannot revise history.** A row that already has an outcome is never
touched again, so re-running after a data correction cannot quietly improve
last month.

Re-predicting a fixture that has *not* been played replaces the earlier call
rather than adding a second — Thursday's call with fresher form is a better
answer to the same question, not a new question.

## PLEA in plain English

PLEA is the Premier League Elo Algorithm — the rating half of the project.

Every club carries one number. It starts at 1500 for everyone in August 2010, and
then every match nudges it. The idea is that you're rewarded for beating what you
weren't expected to beat.

### The four steps, per match

1. **Work out what should happen.** Compare the two numbers, giving the home side a
   bonus of 65 points first. A big gap means the favourite is expected to take
   nearly all the points.
2. **See what actually happened.** Win = 1, draw = 0.5, loss = 0.
3. **Check the margin.** A 4–0 win counts for more than a 1–0.
4. **Move both numbers.** Whatever one club gains, the other loses. Exactly.

### The example — real numbers from the data

Arsenal are on 1729, Burnley on 1321. That's a 408-point gap, which is enormous.

**Arsenal beat Burnley 2–0 at home.** PLEA expected Arsenal to take 0.938 of the
points here — basically a formality. They won, so they beat expectation by a sliver:

```
Arsenal  1729.3 → 1731.1   (+1.8)
Burnley  1321.0 → 1319.2   (−1.8)
```

Barely moves. Arsenal did what everyone knew they'd do.

**Now flip it — Burnley beat Arsenal 2–0.** Burnley were expected to take 0.122 of
the points. They took all of it:

```
Burnley  1321.0 → 1347.3   (+26.3)
Arsenal  1729.3 → 1703.0   (−26.3)
```

Fifteen times the movement. Same scoreline, same margin — the only difference is
that nobody saw it coming.

**And a draw counts as a bad day for Arsenal:**

```
Arsenal  1729.3 → 1720.5   (−8.8)
Burnley  1321.0 → 1329.8   (+8.8)
```

Arsenal lose points from a draw, because 0.5 is well below the 0.938 expected of
them. Burnley gain from not losing. That's the system working as intended.

### Three extra rules

**Season rollover.** Every August, each club gets pulled 20% of the way back
towards 1500 — because good teams sell players and bad teams buy them, and nobody
stays exactly as they were. Arsenal's 1729 at the end of 2025/26 would start the
next season at 1683.

**New clubs start at 1400.** Promoted sides, or anyone returning after years away
where their old number is stale.

**No crowd, no home bonus.** For the 452 matches played behind closed doors
between June 2020 and May 2021 the 65-point bonus is set to zero, because home
advantage did not shrink over that stretch — it disappeared. Home sides took
0.5022 of the points, against 0.5738 everywhere else.

### Why it works

Notice what PLEA never needs: no league table, no squad list, no transfer fees.
Just who played who and what the outcome was. Yet the ratings it produces are
accurate to 1.3 percentage points — when it says a club should take 35% of the
points from a fixture, they take about 37%.

That's what makes it a good foundation for the model. It compresses sixteen years
of football into one honest number per club.

## Where does −8.8 or +26.3 come from? The formula

Seven steps, for either club in any match:

```
gap      = their_rating − your_rating − home_bonus   (+65 if you are home, −65 if away)
ratio    = 10 ^ (gap / 400)
expected = 1 / (1 + ratio)
actual   = 1 if you won, 0.5 if drew, 0 if lost
surprise = actual − expected
margin   = 1 for one goal or a draw, 1.5 for two, more above that
change   = 20 × margin × surprise
```

Then `your new rating = old + change`, and the other club gets the same amount
subtracted.

The two constants are the only magic. **400** sets the scale: 400 rating points
means ten times stronger. **20** is K, the volume knob — one full unit of surprise
moves you 20 points.

### Real example — Arsenal 3–1 Manchester City

Arsenal on 1709 at home, Manchester City on 1733.

**From Arsenal's point of view:**

```
1733 − 1709 − (+65 home)   = −41
10 ^ (−41 / 400)           = 0.79×
1 / (1 + 0.79)             = 0.559     expected share of the points
Arsenal won                = 1.0
1.0 − 0.559                = +0.441    the surprise
won by 2 goals             = ×1.50
20 × 1.50 × 0.441          = +13.2

Arsenal  1709 → 1722.2      Manchester City  1733 → 1719.8
```

**From Manchester City's point of view:**

```
1709 − 1733 − (−65 away)   = +41
10 ^ (41 / 400)            = 1.27×
1 / (1 + 1.27)             = 0.441     expected share of the points
Manchester City lost       = 0.0
0.0 − 0.441                = −0.441
lost by 2 goals            = ×1.50
20 × 1.50 × −0.441         = −13.2

Manchester City  1733 → 1719.8      Arsenal  1709 → 1722.2
```

Same match, both sides, and the two changes cancel exactly. That's step 4.

## What the backtest found

The model beats PLEA — but only after being cut down from forty-two columns to
three, and from a random forest to a logistic regression. Almost everything
tried along the way failed, and the failures were more informative than the win.

### How it is tested

Walk-forward. Train on every season before season S, predict S, move to S+1 —
thirteen rounds, 9,880 predictions, none of them made by a model that had seen
a match played after the one it was predicting.

| Contender | What it knows | Accuracy | Log loss | AUC | Calibration |
|---|---|---|---|---|---|
| **`model`** | PLEA, plus both shot gaps | **69.3%** | **0.5821** | 0.734 | 0.0109 |
| `plea_only` | one number: PLEA's expectation | 69.1% | 0.5843 | 0.730 | 0.0106 |
| `home_or_away` | the home and away win rates | 61.8% | 0.6575 | 0.565 | 0.0245 |
| `base_rate` | one number, the win rate | 61.8% | 0.6655 | 0.495 | 0.0132 |

Log loss is the column that matters — accuracy throws away the difference
between "60% sure" and "99% sure", and a model that simply never predicts an
away win still scores 62%.

Compared match by match rather than on the averages:

```
model          vs plea_only   +0.00221  +/- 0.00092    z = +2.39   REAL
home_or_away   vs plea_only   -0.07283  +/- 0.00447    z = -16.28  REAL
base_rate      vs plea_only   -0.08117  +/- 0.00475    z = -17.10  REAL
```

### The one thing that worked: shots on target

PLEA is built from goals, and goals are the lucky part of football. A club
creating far more than it converts carries a rating that is too low, and it
tends to come back.

Testing that directly — does PLEA's error line up with anything? — every
goal-based measure of form came out as noise, and one column did not:

```
form_gf      +0.0107   z = +1.06   noise
form_ga      -0.0080   z = -0.79   noise
form_points  -0.0022   z = -0.21   noise
form_sot     +0.0403   z = +4.01   REAL      <- shots on target
```

Two details matter. It has to be a **gap** — a club's own shot count partly
measures who it happened to play, and only the difference against the opponent
cancels that out. And the effect is small enough that no single season can see
it: scramble the column on one season and the damage is within noise. Only
across thirteen seasons does it show up.

This is most of what expected goals would have given us, from columns that were
already sitting in the data.

### Why the model got smaller

The first version was a random forest reading all forty-two columns. Measured:

| | Log loss |
|---|---|
| logistic regression, 3 columns | **0.5821** |
| PLEA alone | 0.5843 |
| random forest, 42 columns | 0.5866 |
| random forest, the same 3 columns | 0.5892 |

The forest loses to the regression (z = +2.18), and the *same three columns* in
a forest are worse than doing nothing at all (z = −2.19). The relationship here
is smooth — more shots on target than your opponent, better chance of winning.
A regression spends one coefficient on that. A forest has to build the same
curve out of staircase steps, and on a signal this weak the noise in the steps
costs more than the flexibility is worth.

The whole model is now three numbers, which is the other thing a forest could
never give you:

```
elo_expected   0.8070    odds x2.24
shots_gap      0.1322    odds x1.14
sot_gap        0.0811    odds x1.08
```

### Everything that did not work

**Twenty-four of the original twenty-nine columns.** Dropping form, the
opponent's form, the ratios, the season-to-date figures or the fixture details
changed nothing detectable. Dropping PLEA was the only removal that did real
damage (z = −4.5).

**Stakes** — points from the title, from the top four, from safety, and whether
each is still mathematically reachable. Built on the theory that a club already
safe or already relegated stops trying. Only 432 of 9,880 rows have nothing at
stake, those clubs underperform by about one percentage point, and the sharpest
form of the test comes out at z = +0.49. The columns are kept but earn nothing.

**Opponent-adjusted form** — goals scored measured against what those particular
opponents normally concede. The adjusted column turned out to be 92% identical
to the raw one, because Premier League defences all concede between roughly 0.8
and 2.0 a game and that spread is tiny next to the randomness in five matches.
Swapping it in changed nothing (z = −0.07). Reverted.

**PLEA's home advantage.** Carried for three sessions as the obvious next fix:
the parameter is 65, and the home side's share of the points across 6,110
matches directly implies 47.9. Swept properly against out-of-sample log loss,
48 makes the model *worse* — z = −2.79 on the tuning seasons, −1.43 across all
thirteen — and nothing beats 65 by a detectable margin.

The 47.9 was answering a different question. `home_adv` is a knob inside a
feedback loop rather than an estimate of a quantity: the ratings adapt around
whatever it is set to, and the regression downstream re-fits its own intercept
and slope on `elo_expected` anyway, so PLEA being internally biased costs
nothing the model does not simply absorb. A number can be correct and still not
be the number the parameter wants.

**Attack and defence as separate ratings.** PLEA gives a club one number, so a
side that wins 4-3 every week and one that wins 1-0 look identical to it. The
fix is a different architecture entirely: two ratings per club, updated as an
online Poisson regression, with the win probability coming off a scoreline grid
rather than a sigmoid.

```
lambda_home = exp(base + atk_home - def_away + home_boost)
lambda_away = exp(base + atk_away - def_home)
```

It works, in the sense that it produces sensible ratings — Manchester City the
best attack, Arsenal the best defence — and on its own it matches PLEA almost
exactly: 0.58453 against 0.58439, and the two agree on 0.97 of their
predictions.

That is the finding, rather than the disappointment around it. **Two
architectures with nothing in common — Elo on results, Poisson on goals —
arrive at the same answer.** When independent methods converge that tightly,
what is left is not a better method.

Added to the model it looked like a win on the tuning seasons, at z = +2.83,
and evaporated to z = +0.32 on the seasons held back. Swept over learning rate
and home boost and re-chosen on validation, the best setting still came out at
**z = +0.28** across all thirteen. This is exactly what the held-out seasons
exist to catch: on the tuning window alone it would have shipped.

**Shots conceded.** The same idea in feature form — how many more shots a club
gives up than its opponent does, which nothing in the model can otherwise see.
Strong against the residuals (z = -3.71 for the combined matchup) and dead on
arrival out of sample, worse than nothing on the tuning seasons.

**A better August.** Carried as the last untested idea, on the grounds that
matchweeks 1-3 were the largest remaining hole. Measured against the market
properly, they are not:

| | our log loss | market | our deficit | vs our own midseason | market vs its own |
|---|---|---|---|---|---|
| mw 1-3 | 0.5881 | 0.5770 | 0.0111 | +0.0038 | -0.0008 |
| mw 21-30 | 0.5843 | 0.5778 | 0.0065 | 0.0000 | 0.0000 |
| **mw 31-38** | 0.5853 | **0.5690** | **0.0163** | +0.0010 | **-0.0088** |

August costs us 0.0038 against our own midseason standard, over 8% of the rows
— perhaps 0.0003 overall if it were fixed perfectly, which is an order of
magnitude below what the paired test can see. And the model is not even
over-confident there: the slope of outcomes on predictions is 0.989 in
matchweeks 1-3, closer to perfect than midseason's 0.945.

Both knobs were swept anyway. `carry_over` is best at its current 0.80 and
`promoted_elo` at its current 1400, on the tuning seasons and on all thirteen;
every alternative tried is worse.

**The run-in is the real hole, and it is not ours.** In matchweeks 31-38 our
log loss is flat against our own midseason standard while the market's improves
by 0.0088. They are not beating us there because we degrade. They are beating
us because they get better — which is what team news looks like from the
outside.

### How far from the ceiling?

Bet365's closing odds, scored on the same 9,880 rows with the bookmaker's
margin stripped out, reach **0.5733**. That is the honest ceiling — a number
set by people who know the team news.

```
ignorance  0.6655 ──────────────────────────────────────► 0.5733  market
                  ├───────────────────────────────────┤
                            the model covers 90%
```

The gap is not spread evenly. From November to March we are level with the
market; almost all of the deficit sits in matchweeks 1–3, where PLEA's ratings
are stale after a summer of transfers, and 31–38, where the likeliest
explanation is team news rather than motivation — who is rested, injured, or
being saved for a cup final. That is information we cannot get.

### What works, and what the calls look like

Calibration is good: sorted into ten buckets by confidence, what the model
promises is close to what happens.

| Predicted | Actually won |
|---|---|
| 9.8% | 10.5% |
| 26.4% | 25.6% |
| 45.5% | 43.8% |
| 73.4% | 73.6% |

Rejoining both perspectives gives the right verdict on **62.2% of 4,940
matches**, and not one fixture had the home and away probabilities adding to
more than 1 — the model never contradicted itself.

### Next

Seven ideas tried, one worked. That is not a run of bad luck — it is what the
evidence has been saying for a while, and the attack/defence result is the
clearest statement of it yet: a completely independent architecture, given a
fair hearing and proper tuning, lands on the same number.

**The remaining gap to the market is mostly team news.** Who is rested, injured,
or being saved for a cup final, known an hour before kickoff. It is why the
deficit sits almost entirely in matchweeks 1–3 and 31–38 and vanishes in
midwinter. No re-encoding of results will recover it, because it is not in the
results.

Eight ideas tried, one worked. The list is now empty, and that is a result
rather than a stopping point: the remaining deficit has been traced to a
specific window, matchweeks 31-38, and to a specific cause, the market getting
better there rather than us getting worse. That is team news, and it is not in
the results.

What would actually move the number, in order of how unlikely it is to be
available:

1. **Team sheets** — who is fit, rested, or being saved for a cup final,
   known an hour before kickoff. This is the whole remaining gap.
2. **Expected goals from a usable source.** FBref sits behind a bot
   challenge and Understat's robots.txt disallows crawling, so neither is
   available to us. Shot counts recover most of it and already went in.

The method that repaid the effort every time is the same one: ask **what the
model currently gets wrong**, check which columns line up with those mistakes,
and only then build. It found shots on target, and it correctly called seven
failures before the work was spent on them.

## Edge cases in the training table

Football data has gaps that aren't really gaps — a club playing its first ever
match genuinely has no recent form, and no amount of cleaning invents one. These
are the places the table would otherwise be blank, and what goes there instead.

| Edge case | Rows | How it's handled |
|---|---|---|
| Kickoff time missing before 2019/20 | 6,840 | **Column dropped.** A weak feature whose absence perfectly identified older seasons |
| Points per game, early season | 320 blank, ~3,200 noisy | **Blended toward last season:** `(played × this_season + 5 × last_season) ÷ (played + 5)` |
| Promoted club has no last season | — | Prior of **1.07 ppg** — what promoted clubs actually average, over 65 cases |
| League position, matchweek 1 | 320 | **Last season's finish.** Promoted clubs get **15th**, their real median |
| Rest days at a season's first match | 320 | **Capped at 14 days.** Ninety days off isn't ten times more rested than nine |
| Rest days gap | 326 | Falls out once both sides are filled |
| A club's first match ever | 41 | **League-average form:** 1.40 goals for, 1.40 against, 12.74 shots, 4.89 on target, 5.30 corners, 0.50 points |
| Finishing rate ÷ zero shots on target | 1 | League-average finishing, rather than infinity |
| Matches played to empty stadiums | 904 | **Flagged, not deleted** — `crowd = 0` |

Two ideas run through all of it.

**When you don't know, fall back to the best available prior — and keep `played`
and `matchweek` in the table so the model can see how much evidence backs each
estimate.** A points-per-game figure from two matches and one from thirty look
identical in isolation; those two columns are what let the model tell them apart.

**Never fill a gap with zero.** Zero shots is a real value that means something
different from "we don't know", and blurring the two teaches the model something
false.

The covid row is the one exception to filling anything in, and it's deliberate.
Deleting 452 mid-season matches would punch holes in the rolling-form windows of
the matches either side, corrupting rows that were never affected. Flagged, the
model can learn "when `crowd` is 0, ignore home advantage" — and since every
future match has a crowd, that branch simply never fires when predicting.

The result: no blanks and no infinities anywhere in the table. That keeps the
model choice open rather than letting the data force it.

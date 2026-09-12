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
python scripts/build_dataset.py
python scripts/train_model.py
```

The first script downloads the results, checks them, and builds the ratings and
the training table. The second grades the model against three baselines on
seasons it was never trained on, then trains the one that gets kept.

`notebooks/plea_ratings.ipynb` and `notebooks/feature_table.ipynb`
have the tables and charts; `docs/model-workflow.html` explains how it fits together.

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

Two ideas left, and after four straight negative results the honest prior is
that neither will work either:

1. **Separate attack and defence ratings.** PLEA gives a club one number, so a
   side that wins 4–3 every week and one that wins 1–0 look identical to it —
   despite being very different match-ups. This is the only remaining idea that
   adds a genuinely new *kind* of information rather than re-encoding what the
   rating already holds.
2. **A better August.** Matchweeks 1–3 are the largest remaining hole, and
   PLEA's flat 20% pull toward 1500 each summer is a blunt instrument.

Worth saying plainly: of six things tried, one worked. The reliable move has
been to ask **what PLEA currently gets wrong** and check which columns line up
with those mistakes — that question found shots on target, and it correctly
predicted the failure of the other five before the effort was spent.

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

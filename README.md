# Premier League Predictor

A model that looks at an upcoming Premier League fixture and says whether a team
will win it — yes or no — along with how confident it is.

It works from two things: a rating for every club, built by replaying every match
since 2010, and how each side has been playing lately. Those feed a model trained
on sixteen seasons of results, which returns a probability for each team and one call.

The ratings, the training table and the model are all built, and the model has
been tested against thirteen seasons it never saw. The result was not the one
we wanted — see [What the backtest found](#what-the-backtest-found).

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

### Two extra rules

**Season rollover.** Every August, each club gets pulled 20% of the way back
towards 1500 — because good teams sell players and bad teams buy them, and nobody
stays exactly as they were. Arsenal's 1729 at the end of 2025/26 would start the
next season at 1683.

**New clubs start at 1400.** Promoted sides, or anyone returning after years away
where their old number is stale.

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

The honest answer: **the model is not yet better than the ratings it is built
from.** It is not worse either. It is the same, and that is a real result rather
than a bug to be fixed by trying harder.

### How it was tested

Walk-forward. Train on every season before season S, predict S, move to S+1 —
thirteen rounds, 9,880 predictions, none of them made by a model that had seen
a single match played after the one it was predicting. Three bars to clear:

| Contender | What it knows | Accuracy | Log loss | Brier | AUC | Calibration |
|---|---|---|---|---|---|---|
| `plea_only` | one number: PLEA's expectation | 68.8% | **0.5855** | 0.2004 | 0.729 | 0.0105 |
| `forest` | the whole table | 68.8% | 0.5866 | 0.2006 | 0.728 | 0.0104 |
| `home_or_away` | the home and away win rates | 61.8% | 0.6575 | 0.2325 | 0.565 | 0.0245 |
| `base_rate` | one number, the win rate | 61.8% | 0.6655 | 0.2363 | 0.495 | 0.0132 |

Log loss is the column that matters — accuracy throws away the difference
between "60% sure" and "99% sure", and a model that simply never predicts an
away win still scores 62%.

### Is the gap real?

No. Comparing the two models match by match rather than on their averages:

```
forest vs plea_only    -0.00102  +/- 0.00137    z = -0.75   noise
home_or_away           -0.07191  +/- 0.00449    z = -16.00  REAL
base_rate              -0.07997  +/- 0.00477    z = -16.76  REAL
```

The test has no trouble seeing the baselines lose by a mile. It sees nothing
between the forest and PLEA. Every hyperparameter tried — leaf sizes from 10 to
200, three settings of `max_features` — landed within 0.012 log loss of every
other, and logistic regression and gradient boosting both landed there too. The
ceiling is not the model.

### Why

Scrambling one column at a time and measuring the damage says it plainly:

```
elo_expected      0.0616      <- everything
elo_gap           0.0213      <- the same information, restated
opp_elo           0.0055
own_elo           0.0039
is_home           0.0030
everything else  <=0.0028     <- 22 of 40 columns cost nothing at all
```

The diagnosis is that **form is not opponent-adjusted**. `form_gf = 1.4` means
something completely different depending on whether the last five matches were
against Manchester City or against Burnley. PLEA already knows how good a club
is, so the only genuinely new content in a raw five-match average is *who they
happened to play* — which is schedule noise, not skill. The model correctly
declines to use it.

That is a flaw in the features, not in the model. The 25 form and season columns
were built to describe a club; they mostly describe its fixture list.

### What does work

PLEA is well calibrated, and so is the forest built on it. Across the thirteen
test seasons, sorted into ten buckets by confidence:

| Predicted | Actually won | Gap |
|---|---|---|
| 10.9% | 12.0% | −1.2 |
| 22.3% | 24.2% | −1.9 |
| 38.2% | 38.1% | +0.1 |
| 51.8% | 52.4% | −0.7 |
| 75.6% | 73.2% | +2.4 |

When it says 38%, it happens 38% of the time. Rejoining the two perspectives
into one verdict gives the right call on **61.9% of 4,940 matches**, and there
was not a single fixture where the home and away probabilities added to more
than 1 — the model never contradicted itself.

### Stakes: a negative result

The first attempt at closing the May gap was **stakes** — points from the title,
from the top four, from safety, and whether each is still mathematically
reachable. Eleven columns, on the theory that a club already safe or already
relegated stops trying.

**It did not work.** Log loss went from 0.5867 to 0.5866. The theory itself
turns out not to be in the data:

* only **432 of 9,880** rows have nothing at stake — it is a rare situation
* those clubs underperform PLEA by about **one percentage point**
* the sharpest version of the test, "I still care and they do not" against the
  reverse, comes out at **z = +0.49**

So the May deficit is probably not motivation. The likelier explanation is team
news — Bet365 knows the starting eleven an hour before kickoff, who is rested,
injured, or being saved for a cup final. That is information we cannot get, and
it is a wall rather than a puzzle.

The columns are kept. They are leak-free, cost nothing to carry, and another
season of data may yet push them over the line — but they earn nothing today.

### Next

Three things worth trying, in order of how much they should move the number:

1. **Opponent-adjusted form.** Goals scored relative to what those specific
   opponents usually concede. This is the one that addresses the diagnosis
   above, and the only one on this list that adds information PLEA lacks.
2. **Separate attack and defence ratings.** PLEA collapses a club into a single
   number, so a side that wins 4–3 every week and one that wins 1–0 every week
   look identical to it — despite being very different match-ups.
3. **PLEA v2.** `home_adv` is set to 65 but measures ~47.9 in the data, and the
   era breakdown is stark: 55 points (2010–16), 58 (2016–20), **−8 (2020/21)**,
   43 (2021–26). Frozen deliberately until the feature work is done, so that one
   change is not being evaluated through another.

For scale: bookmakers land somewhere near 0.55 log loss on the same question.
At 0.586 the gap left to close is real but not enormous, and roughly half of a
football match is genuinely not predictable from anything.

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

# Premier League Predictor

A model that looks at an upcoming Premier League fixture and says whether a team
will win it — yes or no — along with how confident it is.

It works from two things: a rating for every club, built by replaying every match
since 2010, and how each side has been playing lately. Those feed a model trained
on sixteen seasons of results, which returns a probability for each team and one call.

Right now the ratings and the training table are built. The model itself isn't trained yet.

## Running it

```bash
pip install -e ".[dev]"
python scripts/build_dataset.py
```

Downloads the results, checks them, builds the ratings and the training table.
`notebooks/plea_ratings.ipynb` and `notebooks/feature_table.ipynb`
has the tables and charts; `docs/model-workflow.html` explains how it fits together.

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

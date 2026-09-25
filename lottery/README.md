# Lottery Laboratory

Draw-history analysis for Powerball and Mega Millions, built to answer one
question honestly: **can you tell which numbers are likely to come up?**

The answer is no, and this package is designed to let you verify that yourself
rather than take anyone's word for it. What it *does* find is the one real edge
in lottery play, which is not about winning more often.

**Status: complete and tested (53 tests).**

```bash
python scripts/lottery_report.py
python scripts/lottery_report.py --game mega_millions --era all
python scripts/lottery_report.py --ticket "38 43 52 61 67" --jackpot 1.5e9
```

---

## The two findings

### 1. No number is hot. Not one.

Across 1,391 Powerball drawings of the current era:

| test | result |
|---|---|
| white balls uniform | chi2 = 78.90, 68 dof, **p = 0.17** |
| Powerballs uniform | chi2 = 23.30, 25 dof, **p = 0.56** |
| serial independence (draw *t* vs *t+1*) | chi2 = 0.26, 1 dof, **p = 0.61** |

The hottest number is 21 at 124 appearances against 100.8 expected — a 23%
excess that looks like a signal. It is not. **A fair machine's hottest number
averages 124.5.** The observed leader is slightly *below* what randomness
predicts (p = 0.55).

That gap is the whole trick. Looking at 69 numbers, picking the largest, and
judging it against the average of a *single* number is a multiple-comparisons
error, and it is the basis of every hot-number system ever sold.
`expected_extremes` simulates the null distribution of the maximum so the
leader gets measured against the right yardstick.

### 2. You can't change your odds. You can change your payout.

The jackpot is **pari-mutuel** — split equally among everyone holding the
winning combination. Players do not pick uniformly: they pick birthdays, lucky
numbers, and lines drawn down the playslip. So some combinations are held by
thousands of tickets and some by almost none.

At a $1.5B jackpot with 300M tickets sold:

| ticket | crowd score | share kept | expected return |
|---|---|---|---|
| `03 07 12 21 28` (birthdays) | 10.98x | 8.9% | **$0.34** |
| `05 10 15 20 25` (playslip line) | 14.37x | 6.8% | $0.31 |
| `38 43 52 61 67` (all high) | 0.23x | 89.2% | **$1.64** |

Same probability of winning — *exactly* the same, to the last digit. Nearly
**5x difference in expected value**, purely from which numbers you write down.

Those multiples come from the heuristic prior described below, so treat their
magnitude as directional; the ordering is robust to any sane parameter choice.

Both are still negative. That is the honest bottom line, and the module says so
in its own output.

---

## Layout

```
lottery/
    games.py        Matrices, historical eras, exact combinatorial odds
    history.py      Draw records, parsing, era assignment, validation
    sources.py      Fetch from the NY State open-data portal
    randomness.py   Chi-square, extremes, gaps, serial independence
    popularity.py   How the crowd picks, and what splitting costs you
    value.py        Expected value with cash option, tax, and splitting

scripts/lottery_report.py   CLI; caches history in data/lottery_history.db
tests/test_lottery.py       53 tests
```

## Design principles

**Eras are not optional.** Powerball has used three matrices since 2009;
Mega Millions five since 2002. Pooling across a change is the single most
common error in lottery analysis — every number above the old maximum looks
"cold" because it did not physically exist, and every number below it looks
"hot" because it had more chances. `validate_draws` refuses any draw containing
a number its era cannot produce, and all 4,511 published drawings pass.

**Odds are computed, not copied.** Every figure comes from `math.comb`, and
the tests assert all nine Powerball tiers against the odds printed on a real
ticket, including the 1-in-24.87 overall.

**Stdlib only.** Per `requirements-web.txt`, no numpy or scipy. The chi-square
p-value is the regularised incomplete gamma function implemented here, verified
exact to five decimals against published critical values from 1 to 100 dof.

**Heuristics are never shown as probabilities.** Exactly as in
`poker/strength.py`. Lotteries do not publish per-combination ticket sales, so
`PopularityModel` is a documented prior, not a measurement. It reports a
*relative crowd score* ("2.3x more popular than average") and never a
probability. `calibration_notes()` describes how to replace it with real
numbers if per-draw winner counts are ever ingested.

**The convexity is not glossed over.** Expected jackpot share with Poisson
co-winners is `(1 - e^-λ)/λ`, not `1/(1+λ)`. The naive form understates your
share; there is a test asserting the difference.

---

## What this is not

Nothing here improves your chance of winning, because nothing can. The draw is
a physical machine with no memory, and the tests in this package are the
evidence for that claim rather than an assertion of it.

The expected-value calculator can report a positive number at extreme
jackpots. That is a statement about averages over millions of tickets, not
about yours: the median outcome of buying a ticket is losing its price, and it
stays that way at every jackpot. This is a tool for understanding probability,
not a system for making money.

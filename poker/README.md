# Poker Laboratory

An interactive No-Limit Texas Hold'em game whose mathematics are visible while
you play. You make the decisions; the application exposes the reasoning
underneath them.

**Status: Milestone 1 complete.**

Play it at `/poker` once the server is running:

```bash
python -m uvicorn backend.main:app --reload
```

---

## Design principles

**Never fabricate a probability.** Every number on screen is exact arithmetic
on observed game state. Quantities that require estimation — equity, outs,
opponent ranges, EV — are reported as *not yet available* with the milestone
that will provide them, rather than approximated. A plausible-looking fake
probability is worse than no probability in a tool meant to build correct
intuition.

**Heuristics are never shown as probabilities.** `strength.py` produces a 0-1
score used *only* to make the AI act plausibly. It is deliberately isolated
from anything the player sees, and its docstring says so.

**Stdlib only in the engine.** `requirements-web.txt` deliberately excludes
numpy/scipy so the web tier stays lean. The poker engine therefore adds no
dependency to it. Numeric libraries belong in the offline simulation lab
(Milestone 10), never in the request path.

**Poker stays separate from sports.** No shared abstraction has been created.
A `decision_theory/` package would be premature until there are genuinely
shared implementations — the existing Bayesian and calibration code in this
repo is basketball-specific and does not generalise.

---

## Layout

```
poker/
    cards.py        Card, Rank, Suit, Deck (seeded → every hand is replayable)
    evaluator.py    5/6/7-card evaluation with exhaustive tiebreakers
    strength.py     Heuristic score for AI decisions ONLY — never displayed
    decision.py     Exactly-derived decision maths (pot odds, SPR, made hand)
    session.py      Stacks, button rotation, hand history across many hands
    engine/
        state.py    Data model: players, pots, streets, action records
        rules.py    Legal actions, minimum raises, reopening rules
        hand.py     The hand state machine: deal → betting → showdown → payout
    shuffle.py      Physical shuffle models: riffle, strip, cut, overhand, wash
    shuffle_analysis.py  Exact mixing theory + empirical randomness tests
    ranges.py       Hand ranges as sets: combos, blockers, inclusion-exclusion
    fourier.py      Fourier analysis on groups; why a cut is not a shuffle
    opponents/
        base.py     Opponent protocol + behavioural parameter profiles
        heuristic.py Profile-driven policy

backend/routes/poker.py   API under /api/poker
frontend/poker/           Vanilla JS table UI at /poker
```

## Rules the engine gets right

These are the cases poker engines most often get wrong, each covered by tests:

- Heads-up blinds: the **button posts the small blind**, acts first preflop,
  and acts last on every later street.
- The big blind's **option** to raise a limped pot.
- The **minimum-raise ladder**: a raise must increase the bet by at least the
  previous increment.
- A **short all-in does not reopen betting** — players who already acted may
  only call or fold.
- **Side pots** by contribution level, with folded players' chips staying in
  the pot but winning nothing.
- **Uncalled bets are returned**, not won at showdown.
- **Split pots** on exact ties, including when the board plays.
- **Odd chips** go to the first winner left of the button.

## Shuffle laboratory

`Deck` shuffles with `random.shuffle` — uniform by construction, and nothing
like a casino. `shuffle.py` models what hands actually do, and
`shuffle_analysis.py` measures the result.

```bash
python scripts/shuffle_report.py --all
```

The riffle is the Gilbert-Shannon-Reeds model, so the exact theory applies.
Bayer & Diaconis (1992) give the probability of an arrangement after `m`
riffles as `C(2^m + n - r, n) / 2^(mn)` for `r` rising sequences, and since the
number of arrangements with `r` rising sequences is an Eulerian number, total
variation distance is a 52-term sum computable in exact integer arithmetic.
**The published table is reproduced to three decimals for m = 1..10**, and
`riffles_needed(0.5)` derives the famous answer rather than asserting it:

| riffles | 4 | 5 | 6 | **7** | 8 | 10 |
|---|---|---|---|---|---|---|
| distance from random | 1.000 | 0.924 | 0.614 | **0.334** | 0.167 | 0.043 |

The fall is abrupt. Four riffles is not "most of the way" to seven.

### What that says about real procedures

| procedure | rising seqs | order preserved | flagged non-random |
|---|---|---|---|
| single riffle + cut | 3.0 | 75.6% | 100% of runs |
| 2 riffles + cut | 5.0 | 63.0% | 100% |
| **casino standard** (riffle, riffle, strip, riffle, cut) | 12.8 | 54.5% | **100%** |
| 7 riffles | 24.6 | 52.1% | 45% |
| 10 riffles | 26.3 | 51.9% | 0% |
| casino deck change (wash first) | 26.7 | 51.8% | 10% |

A uniformly shuffled deck averages ~26.5 rising sequences and 50% order
preservation. **The standard poker-room hand shuffle is detectably non-random
in every run** — three riffles is not close to enough. What rescues a real
casino is the wash on deck changes, which is uniform by construction, plus
automatic shufflers on most modern tables.

Three findings worth keeping:

- **A cut adds exactly zero randomness.** It is a rotation: no card moves
  relative to any other in cyclic order. Its purpose is procedural — denying
  the dealer knowledge of the top card — not statistical. `order_preservation`
  de-rotates before measuring for this reason; without that correction a cut
  drags the metric to 50% and makes the *worst* procedure look perfect.
- **Order preservation must be read two-sided.** One overhand scores 0.38, two
  score 0.65. Reversing a block flips every pair spanning it, so distance from
  0.5 is the signal, not the raw value.
- **A single p-value is one draw from Uniform[0,1].** Reporting one would brand
  the provably-uniform wash procedure as biased about one run in twenty, so
  `significant_fraction` replicates instead. Same lesson as
  `lottery.randomness.expected_extremes`.

### Seeding

`cards.py` seeds from `randrange(2**63)`, so it reaches 9.2×10¹⁸ of the
8.1×10⁶⁷ possible deck orders — a fraction of 1.1×10⁻⁴⁹. For a replayable
learning tool that is the right trade and no player could exploit it. For
anything dealing for money it is exactly why regulated shufflers use hardware
entropy.

## Ranges as sets

What poker calls "combinatorics" is applied set theory, and `ranges.py` says so
directly. A combo is a set of two cards, a range is a set of combos, and every
range operation is a set operation — including `~` for complement, so De
Morgan's laws hold and are tested.

```bash
python scripts/poker_math.py --ranges
python scripts/poker_math.py --outs "Jh Th" "9h 8s 2h"
```

**Blockers are set difference.** `AA` has C(4,2) = 6 combos. Hold one ace and
`AA.remove_dead(As)` leaves 3 — your opponent is literally half as likely to
hold aces as the raw count suggests. `KK` is untouched. That asymmetry is the
whole idea, and it falls out of `-` rather than out of a rule.

**Outs are a union, and unions need inclusion-exclusion.** This is where the
money is:

| J♥T♥ on 9♥8♠2♥ | count |
|---|---|
| flush outs | 9 |
| straight outs | 8 |
| **in both sets** | **2** |
| naive sum | 17 ← wrong |
| \|A ∪ B\| = 9 + 8 − 2 | **15** |

The classic 15-out combo draw, 31.9% to hit on the next card. Adding 9 + 8
double-counts 7♥ and Q♥. `draw_analysis` computes each term from the actual card
sets, so the identity is verified rather than remembered — and it is asserted on
five different boards in the tests.

## Complex conjugation: where it genuinely belongs

Complex numbers are the standard tool for proving how fast a shuffle mixes, and
`fourier.py` implements the case that can be done exactly.

```bash
python scripts/poker_math.py --cuts
```

A cut is a rotation, so repeated cutting is a random walk on the **cyclic group
ℤ₅₂** — abelian, so its characters are just roots of unity `χ_m(j) = ω^(jm)`.
The Diaconis–Shahshahani upper bound lemma then gives

```
4·‖P*ᵏ − U‖²  ≤  Σ_{m≠0} |P̂(m)|^{2k}        where |z|² = z · conj(z)
```

The conjugation is load-bearing, not decorative: multiplying by the conjugate is
what turns an oscillating complex amplitude into a magnitude that can decay.
Character orthogonality is stated with it too.

Measured on the real `cut()` from `shuffle.py`:

| cuts | distance from uniform **rotation** | distance from a **shuffled deck** |
|---|---|---|
| 1 | 0.571 | 1.000000 |
| 5 | 0.237 | 1.000000 |
| 10 | 0.088 | 1.000000 |
| 25 | 0.005 | 1.000000 |

The left column converges. **The right column never moves.** Cutting reaches
only 52 of the 52! arrangements, so `cuts_never_mix` returns exactly
`1 − 52/52!`, which is 1.0 to every digit a float carries. Twenty-five cuts
leave the deck uniformly *rotated* and perfectly ordered — the rigorous version
of the informal claim above that a cut adds no randomness.

The bound is checked against direct convolution at every step from 1 to 25.
Two independent routes — characters and conjugation on one side, brute-force
convolution on the other — agreeing is real evidence that both are right.

**Where conjugation does *not* belong:** hand evaluation and equity. Riffles
live in the non-abelian S₅₂, where the same lemma holds with `Tr(A A*)` for the
conjugate transpose — conjugation survives, it just grows a transpose. But
implementing S₅₂'s representations is research-scale, and unnecessary here:
`shuffle_analysis` already gets the exact riffle answer combinatorially. This
module does the abelian case properly rather than the non-abelian case badly.

## Testing

```bash
python -m pytest tests/test_poker_*.py -q          # ~120 tests, under 4s
POKER_EXHAUSTIVE=1 python -m pytest tests/test_poker_evaluator.py
```

The exhaustive test classifies **all 2,598,960 five-card hands** and checks
each category against its known frequency (40 straight flushes, 624 quads,
3744 full houses, 5108 flushes, 10200 straights, …). It is the strongest
correctness check available for the evaluator and is skipped by default only
because it takes ~2 minutes. **Run it before trusting any evaluator change.**

## Roadmap

Revised 2026-08-06: a **Fundamentals Trainer** was inserted as the new M2.
The engine is correct, but correctness is useless to a player who does not yet
know what "preflop" means. Teaching the vocabulary comes before teaching the
mathematics — "equity 67%" is noise until you know what a flop is.

| # | Milestone | Status |
|---|---|---|
| 1 | Playable heads-up Hold'em (incl. evaluator + pot mechanics) | **done** |
| 2 | **Poker Fundamentals Trainer** — teach the rules during play | **next** |
| 3 | Equity laboratory: Monte Carlo + exact enumeration | |
| 4 | Bayesian range trainer | |
| 5 | EV / decision trainer | |
| 6 | Opponent modelling | parameter surface exists; policies are static |
| 7 | Post-hand replay and analysis | |
| 8 | Player statistics | |
| 9 | Simulation laboratory | |
| 10 | Game theory / CFR experiments | |
| — | **Shuffle laboratory** (physical models + exact mixing theory) | **done** |

### Milestone 2 design intent

Contextual and adaptive, not a textbook. The trainer explains a concept the
first time the player meets it, then gets out of the way:

> **Flop** — the first three community cards. Everyone shares them; you
> combine them with your two private cards to make your best five-card hand.

It should track which concepts have been encountered, so explanations retire
themselves. It must not touch the engine — the teaching layer sits above a
component that already passes its tests.

### Known limitations at Milestone 1

- **Sessions are in-memory.** A server restart loses them. Persistence is
  Milestone 7; every payload is already provenance-shaped (seeded deck, full
  action log, engine version) so that becomes a save call, not a redesign.
- **The evaluator runs at roughly 20k evaluations/second.** Fine for one
  showdown per hand, but Milestone 3 will need it optimised before Monte Carlo
  at scale is comfortable.
- **The opponent is intentionally simple.** It compares heuristic strength to
  pot odds and nothing more. It does not model the player, bluff coherently,
  or balance its ranges.

---

*This is a tool for learning probability and decision-making. Nothing here is
a system for winning money, and no strategy it teaches should be understood as
one.*

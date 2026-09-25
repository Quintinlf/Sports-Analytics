"""Tests for the lottery analysis package.

The published-odds test is the important one: it checks the combinatorics
against figures the lottery itself publishes, so an error in the matrix
definitions cannot pass silently.
"""
from __future__ import annotations

import math
import random
from datetime import date

import pytest

from lottery.games import (
    MEGA_MILLIONS,
    POWERBALL,
    Matrix,
    get_game,
)
from lottery.history import Draw, parse_record, parse_records, validate_draws
from lottery.popularity import (
    PopularityModel,
    expected_cowinners,
    expected_share,
)
from lottery.randomness import (
    chi_square_pvalue,
    expected_extremes,
    special_counts,
    serial_independence_test,
    uniformity_test,
    white_counts,
)
from lottery.value import breakeven_jackpot, ticket_ev


# ---------------------------------------------------------------------------
# Matrix and odds
# ---------------------------------------------------------------------------


def test_powerball_jackpot_odds_match_published():
    assert POWERBALL.matrix.total_combinations == 292_201_338


def test_mega_millions_jackpot_odds_match_published():
    assert MEGA_MILLIONS.matrix.total_combinations == 290_472_336


@pytest.mark.parametrize(
    "matched_white,matched_special,published",
    [
        (5, True, 292_201_338.00),
        (5, False, 11_688_053.52),
        (4, True, 913_129.18),
        (4, False, 36_525.17),
        (3, True, 14_494.11),
        (3, False, 579.76),
        (2, True, 701.33),
        (1, True, 91.98),
        (0, True, 38.32),
    ],
)
def test_powerball_tier_odds_match_published(matched_white, matched_special, published):
    """Every tier, against the odds printed on the back of a real ticket."""
    odds = POWERBALL.matrix.tier_odds(matched_white, matched_special)
    assert odds == pytest.approx(published, abs=0.01)


def test_overall_odds_match_published():
    matrix = POWERBALL.matrix
    total = sum(
        matrix.tier_probability(t.matched_white, t.matched_special)
        for t in POWERBALL.prize_tiers
    )
    assert 1 / total == pytest.approx(24.87, abs=0.01)


def test_tier_probabilities_are_a_partition():
    """Summed over every outcome, probabilities must equal exactly 1."""
    matrix = POWERBALL.matrix
    total = sum(
        matrix.tier_probability(white, special)
        for white in range(matrix.white_count + 1)
        for special in (True, False)
    )
    assert total == pytest.approx(1.0, abs=1e-12)


def test_matrix_rejects_impossible_shape():
    with pytest.raises(ValueError):
        Matrix(white_count=6, white_max=5, special_max=10)


def test_get_game_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown game"):
        get_game("keno")


# ---------------------------------------------------------------------------
# Eras
# ---------------------------------------------------------------------------


def test_eras_are_contiguous_and_ordered():
    for game in (POWERBALL, MEGA_MILLIONS):
        for earlier, later in zip(game.eras, game.eras[1:]):
            assert earlier.end is not None
            assert earlier.end < later.start, f"{game.name} eras overlap"


def test_era_lookup_picks_the_right_matrix():
    """The 2015 Powerball change is the boundary every analysis depends on."""
    assert POWERBALL.era_for(date(2015, 10, 3)).matrix.white_max == 59
    assert POWERBALL.era_for(date(2015, 10, 7)).matrix.white_max == 69


def test_era_lookup_rejects_dates_before_coverage():
    with pytest.raises(ValueError, match="no defined era"):
        POWERBALL.era_for(date(1999, 1, 1))


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def test_parse_powerball_packs_special_last():
    draw = parse_record(
        POWERBALL,
        {
            "draw_date": "2026-08-19T00:00:00.000",
            "winning_numbers": "10 21 58 61 64 17",
            "multiplier": "2",
        },
    )
    assert draw.whites == (10, 21, 58, 61, 64)
    assert draw.special == 17
    assert draw.multiplier == 2


def test_parse_mega_millions_uses_separate_field():
    draw = parse_record(
        MEGA_MILLIONS,
        {
            "draw_date": "2026-08-18T00:00:00.000",
            "winning_numbers": "05 19 30 38 59",
            "mega_ball": "12",
        },
    )
    assert draw.whites == (5, 19, 30, 38, 59)
    assert draw.special == 12


def test_parse_rejects_wrong_token_count():
    with pytest.raises(ValueError, match="expected 6 packed numbers"):
        parse_record(
            POWERBALL,
            {"draw_date": "2026-08-19T00:00:00.000", "winning_numbers": "1 2 3"},
        )


def test_parsed_records_come_back_sorted_oldest_first():
    records = [
        {"draw_date": "2026-08-19T00:00:00.000", "winning_numbers": "10 21 58 61 64 17"},
        {"draw_date": "2026-08-15T00:00:00.000", "winning_numbers": "05 08 27 29 63 13"},
    ]
    draws = parse_records(POWERBALL, records)
    assert [d.draw_date for d in draws] == [date(2026, 8, 15), date(2026, 8, 19)]


def test_validate_catches_number_impossible_for_its_era():
    """A 69 in 2014 is impossible: the pool only went to 59."""
    draw = Draw(
        game_key="powerball",
        draw_date=date(2014, 6, 1),
        whites=(1, 2, 3, 4, 69),
        special=5,
        era_index=POWERBALL.era_index(date(2014, 6, 1)),
    )
    problems = validate_draws(POWERBALL, [draw])
    assert len(problems) == 1
    assert "outside 1..59" in problems[0].reason


def test_validate_catches_out_of_range_special():
    draw = Draw(
        game_key="powerball",
        draw_date=date(2020, 6, 1),
        whites=(1, 2, 3, 4, 5),
        special=30,
        era_index=POWERBALL.era_index(date(2020, 6, 1)),
    )
    problems = validate_draws(POWERBALL, [draw])
    assert any("outside 1..26" in p.reason for p in problems)


def test_draw_rejects_unsorted_or_duplicate_whites():
    with pytest.raises(ValueError, match="sorted"):
        Draw("powerball", date(2020, 1, 1), (5, 1, 2, 3, 4), 1, 2)
    with pytest.raises(ValueError, match="duplicate"):
        Draw("powerball", date(2020, 1, 1), (1, 1, 2, 3, 4), 1, 2)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statistic,dof",
    [(3.8415, 1), (5.9915, 2), (7.8147, 3), (11.0705, 5), (18.3070, 10), (124.3421, 100)],
)
def test_chi_square_critical_values(statistic, dof):
    """Each of these is the published 5% critical value for its dof."""
    assert chi_square_pvalue(statistic, dof) == pytest.approx(0.05, abs=1e-4)


def test_chi_square_known_closed_forms():
    # Q(1, 1) for 2 dof is exactly e^-1.
    assert chi_square_pvalue(2.0, 2) == pytest.approx(math.exp(-1), abs=1e-9)
    assert chi_square_pvalue(0.0, 5) == pytest.approx(1.0)


def test_chi_square_rejects_bad_input():
    with pytest.raises(ValueError):
        chi_square_pvalue(1.0, 0)
    with pytest.raises(ValueError):
        chi_square_pvalue(-1.0, 1)


def _fair_draws(count: int, matrix: Matrix, seed: int = 7) -> list[Draw]:
    rng = random.Random(seed)
    pool = list(matrix.white_range())
    era_index = len(POWERBALL.eras) - 1
    return [
        Draw(
            game_key="powerball",
            draw_date=date(2020, 1, 1),
            whites=tuple(sorted(rng.sample(pool, matrix.white_count))),
            special=rng.randint(1, matrix.special_max),
            era_index=era_index,
        )
        for _ in range(count)
    ]


def test_uniform_accepts_a_fair_machine():
    matrix = POWERBALL.matrix
    draws = _fair_draws(2000, matrix)
    assert uniformity_test(white_counts(draws, matrix)).p_value > 0.01
    assert uniformity_test(special_counts(draws, matrix)).p_value > 0.01


def test_uniform_detects_a_rigged_machine():
    """Sanity check that the test has power: force ball 1 to appear constantly."""
    matrix = POWERBALL.matrix
    draws = _fair_draws(2000, matrix)
    rigged = [
        Draw(d.game_key, d.draw_date, tuple(sorted({1} | set(d.whites[1:]))), d.special, d.era_index)
        for d in draws
    ]
    assert uniformity_test(white_counts(rigged, matrix)).p_value < 1e-6


def test_counts_are_zero_filled_across_the_pool():
    """A missing category would silently shrink the degrees of freedom."""
    matrix = POWERBALL.matrix
    counts = white_counts(_fair_draws(5, matrix), matrix)
    assert len(counts) == matrix.white_max
    assert set(counts) == set(matrix.white_range())


def test_serial_independence_on_fair_draws():
    matrix = POWERBALL.matrix
    result = serial_independence_test(_fair_draws(1500, matrix), matrix)
    assert result.p_value > 0.01


def test_expected_extremes_finds_fair_maximum_unremarkable():
    """The core anti-hot-number result, on data known to be fair."""
    matrix = POWERBALL.matrix
    counts = white_counts(_fair_draws(1400, matrix), matrix)
    result = expected_extremes(counts, 1400, matrix.white_count, trials=400)
    assert result.max_pvalue > 0.05
    # The hottest number always sits well above the per-number average.
    assert result.simulated_max_mean > result.expected


# ---------------------------------------------------------------------------
# Popularity
# ---------------------------------------------------------------------------


def test_birthday_tickets_are_more_crowded_than_high_tickets():
    model = PopularityModel(POWERBALL.matrix)
    birthdays = model.crowd_score([3, 7, 12, 21, 28])
    high = model.crowd_score([38, 43, 52, 61, 67])
    assert birthdays > 1.0 > high
    assert birthdays / high > 5


def test_arithmetic_sequences_are_penalised():
    model = PopularityModel(POWERBALL.matrix)
    line = model.crowd_score([35, 40, 45, 50, 55])
    scattered = model.crowd_score([35, 41, 46, 52, 57])
    assert line > scattered


def test_thirteen_is_an_asset():
    model = PopularityModel(POWERBALL.matrix)
    assert model.number_weight(13) < model.number_weight(14)


def test_crowd_score_of_average_ticket_is_about_one():
    """Normalisation must centre random tickets on 1.0, or nothing is comparable."""
    model = PopularityModel(POWERBALL.matrix)
    rng = random.Random(99)
    pool = list(POWERBALL.matrix.white_range())
    scores = [model.crowd_score(rng.sample(pool, 5)) for _ in range(3000)]
    assert sum(scores) / len(scores) == pytest.approx(1.0, abs=0.1)


def test_popularity_rejects_malformed_tickets():
    model = PopularityModel(POWERBALL.matrix)
    with pytest.raises(ValueError, match="duplicate"):
        model.raw_score([1, 1, 2, 3, 4])
    with pytest.raises(ValueError, match="expected 5"):
        model.raw_score([1, 2, 3])
    with pytest.raises(ValueError, match="outside"):
        model.raw_score([1, 2, 3, 4, 70])


# ---------------------------------------------------------------------------
# Splitting and expected value
# ---------------------------------------------------------------------------


def test_expected_share_matches_simulation():
    """E[1/(1+K)] for Poisson K, checked against a direct simulation."""
    rng = random.Random(4)
    for lam in (0.2, 1.0, 3.0):
        total = 0.0
        trials = 40_000
        for _ in range(trials):
            # Knuth's Poisson sampler.
            k, p, limit = 0, 1.0, math.exp(-lam)
            while True:
                p *= rng.random()
                if p <= limit:
                    break
                k += 1
            total += 1 / (1 + k)
        assert expected_share(lam) == pytest.approx(total / trials, abs=0.01)


def test_expected_share_is_not_the_naive_reciprocal():
    """1/(1+lambda) understates the share; the convexity is the whole point."""
    assert expected_share(3.0) > 1 / (1 + 3.0)


def test_expected_share_edges():
    assert expected_share(0.0) == 1.0
    assert expected_share(1e-12) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        expected_share(-1.0)


def test_expected_cowinners_scales_with_crowd_and_sales():
    matrix = POWERBALL.matrix
    base = expected_cowinners(1.0, 300_000_000, matrix)
    assert expected_cowinners(2.0, 300_000_000, matrix) == pytest.approx(2 * base)
    assert expected_cowinners(1.0, 600_000_000, matrix) > base


def test_unpopular_numbers_raise_expected_value():
    """Same odds, better payout - the only lever a player actually has."""
    crowded = ticket_ev(POWERBALL, 1.5e9, tickets_sold=300_000_000, crowd_score=13.0)
    lonely = ticket_ev(POWERBALL, 1.5e9, tickets_sold=300_000_000, crowd_score=0.2)
    assert lonely.expected_return > 3 * crowded.expected_return


def test_typical_jackpot_is_negative_ev():
    result = ticket_ev(POWERBALL, 200e6, tickets_sold=30_000_000)
    assert result.expected_profit < 0


def test_expected_value_rises_with_jackpot():
    low = ticket_ev(POWERBALL, 100e6, tickets_sold=20_000_000)
    high = ticket_ev(POWERBALL, 900e6, tickets_sold=20_000_000)
    assert high.expected_return > low.expected_return


def test_ev_rejects_impossible_parameters():
    with pytest.raises(ValueError):
        ticket_ev(POWERBALL, -1, tickets_sold=1000)
    with pytest.raises(ValueError):
        ticket_ev(POWERBALL, 1e9, tickets_sold=1000, cash_ratio=1.5)
    with pytest.raises(ValueError):
        ticket_ev(POWERBALL, 1e9, tickets_sold=1000, tax_rate=1.0)


def test_breakeven_jackpot_is_reachable_when_sales_are_low():
    jackpot = breakeven_jackpot(POWERBALL, tickets_sold=20_000_000, crowd_score=0.5)
    assert jackpot is not None
    result = ticket_ev(
        POWERBALL, jackpot, tickets_sold=20_000_000, crowd_score=0.5
    )
    assert result.expected_profit == pytest.approx(0.0, abs=1e-4)


def test_breakeven_is_unreachable_when_the_crowd_splits_it_back():
    """Heavy sales on a crowded pick can make no jackpot large enough."""
    assert (
        breakeven_jackpot(POWERBALL, tickets_sold=2_000_000_000, crowd_score=20.0)
        is None
    )

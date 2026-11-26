"""Tests for the Wilson score interval."""

import pytest

from evals.v2.stats import wilson_interval


def test_wilson_interval_known_values() -> None:
    lo, hi = wilson_interval(45, 50)
    assert lo == pytest.approx(0.786, abs=1e-3)
    assert hi == pytest.approx(0.957, abs=1e-3)
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


@pytest.mark.parametrize("k,n", [(0, 0), (1, -1), (6, 5), (-1, 5)])
def test_wilson_interval_rejects_bad_input(k, n) -> None:
    with pytest.raises(ValueError):
        wilson_interval(k, n)

import numpy as np
import pytest

from covariance_shift.portfolio import drift, minimum_variance, walk_forward
from covariance_shift.simulation import generate_path


def test_diagonal_portfolio_matches_inverse_variance():
    c = np.diag([1.0, 2.0, 4.0])
    assert np.allclose(minimum_variance(c), np.array([4, 2, 1]) / 7, atol=1e-5)


def test_equal_covariance_and_singular_covariance():
    assert np.allclose(minimum_variance(np.eye(6)), np.full(6, 1 / 6))
    assert np.allclose(minimum_variance(np.ones((6, 6))), np.full(6, 1 / 6))


def test_bad_covariance():
    with pytest.raises(ValueError):
        minimum_variance(np.diag([1.0, -1.0]))


def test_drift_accounting():
    weights, gross = drift(np.array([0.5, 0.5]), np.array([0.1, -0.1]))
    assert gross == pytest.approx(0)
    assert np.allclose(weights, [0.55, 0.45])


@pytest.mark.parametrize("method", ["sample", "ledoit_wolf", "rmt_clip", "equal_weight"])
def test_future_returns_cannot_change_current_allocation(method):
    returns, _ = generate_path(n_assets=6, days=100, change_day=65, seed=5)
    _, before = walk_forward(returns, method, lookback=30, rebalance=10)
    changed = returns.copy()
    changed[50:] = 2 * changed[50:] + 0.001
    _, after = walk_forward(changed, method, lookback=30, rebalance=10)
    for a, b in zip(before, after):
        if a["day"] <= 50:
            assert np.allclose(a["weights"], b["weights"])
            assert a["forecast_variance"] == pytest.approx(b["forecast_variance"])


def test_turnover_uses_drifted_holdings_and_charges_initial_entry():
    returns = np.array([[0.01, -0.01], [-0.01, 0.01], [0.02, -0.02], [0.1, -0.1], [0, 0]])
    daily, allocations = walk_forward(returns, "equal_weight", lookback=3, rebalance=1, cost_bps=10)
    assert allocations[0]["turnover"] == pytest.approx(1)
    assert allocations[1]["turnover"] == pytest.approx(0.1)
    assert daily[0]["net_return"] == pytest.approx(-0.001)
    assert daily[1]["net_return"] == pytest.approx(-0.0001)


def test_costs_do_not_change_weights_but_reduce_wealth():
    returns, _ = generate_path(n_assets=5, days=70, change_day=40, seed=6)
    free, a = walk_forward(returns, "rmt_clip", lookback=20, cost_bps=0)
    paid, b = walk_forward(returns, "rmt_clip", lookback=20, cost_bps=20)
    assert all(np.allclose(x["weights"], y["weights"]) for x, y in zip(a, b))
    assert np.prod([1 + x["net_return"] for x in paid]) < np.prod([1 + x["net_return"] for x in free])

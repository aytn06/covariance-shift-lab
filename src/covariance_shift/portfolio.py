"""Long-only risk allocation and explicit buy-and-hold accounting."""

import numpy as np
from scipy.optimize import minimize

from .estimators import estimate, validate_panel


def minimum_variance(covariance):
    c = np.asarray(covariance, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.isfinite(c).all():
        raise ValueError("Expected a finite square covariance matrix")
    if not np.allclose(c, c.T, atol=1e-12, rtol=1e-8):
        raise ValueError("Covariance must be symmetric")
    scale = np.trace(c) / len(c)
    if scale <= 0 or np.linalg.eigvalsh(c).min() < -1e-8 * scale:
        raise ValueError("Covariance must be positive semidefinite with positive trace")
    c = c / scale + 1e-8 * np.eye(len(c))
    initial = np.full(len(c), 1 / len(c))
    fit = minimize(
        lambda w: w @ c @ w, initial, jac=lambda w: 2 * c @ w,
        method="SLSQP", bounds=[(0, 1)] * len(c),
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                     "jac": lambda w: np.ones_like(w)},
        options={"ftol": 1e-11, "maxiter": 1000},
    )
    if not fit.success:
        raise RuntimeError(f"Minimum-variance solver failed: {fit.message}")
    weights = np.maximum(fit.x, 0)
    weights /= weights.sum()
    # The simplex Frank-Wolfe gap is zero at the convex optimum.
    gradient = 2 * c @ weights
    if weights @ gradient - gradient.min() > 2e-5:
        raise RuntimeError("Minimum-variance solution failed its optimality-gap check")
    return weights


def drift(weights, daily_returns):
    w = np.asarray(weights, dtype=float)
    r = np.asarray(daily_returns, dtype=float)
    if w.shape != r.shape or not np.isfinite(r).all() or np.any(r <= -1):
        raise ValueError("Simple returns must be finite, matching, and greater than -100%")
    if not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1):
        raise ValueError("Weights must be a long-only fully invested portfolio")
    gross = float(w @ r)
    return w * (1 + r) / (1 + gross), gross


def walk_forward(returns, method, lookback=96, rebalance=21, cost_bps=5):
    x = validate_panel(returns)
    if lookback < 3 or lookback >= len(x) or rebalance < 1:
        raise ValueError("Invalid rolling schedule")
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 5000:
        raise ValueError("Cost must be nonnegative and below 5000 bps")
    if np.any(x <= -1):
        raise ValueError("Simple returns must be greater than -100%")
    holdings = np.zeros(x.shape[1])
    daily, allocations = [], []
    for day in range(lookback, len(x)):
        turnover = 0.0
        if (day - lookback) % rebalance == 0:
            # Only observations strictly before the trading day enter the estimate.
            covariance = estimate(x[day - lookback:day], "sample" if method == "equal_weight" else method)
            target = np.full(x.shape[1], 1 / x.shape[1]) if method == "equal_weight" else minimum_variance(covariance)
            # Total buy-plus-sell notional; initial investment from cash costs 1x NAV.
            turnover = float(np.abs(target - holdings).sum())
            holdings = target.copy()
            allocations.append({"day": day, "weights": target.copy(),
                                "forecast_variance": float(target @ covariance @ target),
                                "turnover": turnover})
        holdings, gross = drift(holdings, x[day])
        cost = turnover * cost_bps / 10_000
        net = (1 - cost) * (1 + gross) - 1
        daily.append({"day": day, "gross_return": gross, "net_return": net,
                      "turnover": turnover, "cost_fraction": cost})
    return daily, allocations

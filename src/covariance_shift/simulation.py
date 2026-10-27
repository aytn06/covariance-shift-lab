"""Synthetic returns. No downloaded prices or historical-market claims."""

from dataclasses import dataclass

import numpy as np

SCENARIOS = ("independent", "weak_factors", "factors", "heavy_tails", "regime_shift")


@dataclass
class Sample:
    train: np.ndarray
    future_covariance: np.ndarray


def factor_covariance(n, rng, strength=1.0):
    if n < 2 or strength < 0:
        raise ValueError("Need at least two assets and nonnegative factor strength")
    loadings = rng.normal(0, 0.35, (n, 3))
    loadings[:, 0] = rng.uniform(0.3, 0.9, n)
    raw = strength * (loadings @ loadings.T) + np.diag(rng.uniform(0.3, 0.8, n))
    corr = raw / np.sqrt(np.outer(np.diag(raw), np.diag(raw)))
    sd = rng.uniform(0.007, 0.02, n)
    return corr * np.outer(sd, sd)


def draw_returns(covariance, observations, rng, df=None):
    if observations < 1 or (df is not None and df <= 2):
        raise ValueError("Need positive observations and finite-variance t noise")
    z = rng.normal(size=(observations, len(covariance)))
    if df is not None:
        # Shared radial shock produces multivariate t noise, scaled to covariance C.
        z *= np.sqrt((df - 2) / rng.chisquare(df, observations))[:, None]
    return z @ np.linalg.cholesky(covariance).T


def generate_sample(scenario, n_assets, observations, seed):
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if observations < 4:
        raise ValueError("Need at least four training observations")
    rng = np.random.default_rng(seed)
    strength = 0.12 if scenario == "weak_factors" else 1.0
    old = factor_covariance(n_assets, rng, strength)
    if scenario == "independent":
        old = np.diag(np.diag(old))
    if scenario == "regime_shift":
        new = factor_covariance(n_assets, rng, 1.5) * 1.6
        cut = 2 * observations // 3
        train = np.vstack((draw_returns(old, cut, rng), draw_returns(new, observations - cut, rng)))
        return Sample(train, new)
    df = 5 if scenario == "heavy_tails" else None
    return Sample(draw_returns(old, observations, rng, df), old)


def generate_path(n_assets=24, days=840, change_day=480, seed=1701):
    if not 0 < change_day < days:
        raise ValueError("Change must fall strictly within the path")
    rng = np.random.default_rng(seed)
    old = factor_covariance(n_assets, rng)
    new = factor_covariance(n_assets, rng, 1.5) * 1.6
    returns = np.vstack((draw_returns(old, change_day, rng), draw_returns(new, days - change_day, rng)))
    truth = np.repeat(old[None, :, :], days, axis=0)
    truth[change_day:] = new
    return returns, truth

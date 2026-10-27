"""Transparent baselines and a conservative MP-edge clipping heuristic."""

import numpy as np
from sklearn.covariance import LedoitWolf


def validate_panel(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite observations-by-assets panel, at least 2 by 2")
    if np.any(np.var(x, axis=0, ddof=1) <= 0):
        raise ValueError("Constant assets cannot be converted to correlations")
    return x


def mp_support(q, variance=1.0):
    if not np.isfinite(q) or not np.isfinite(variance) or q <= 0 or variance <= 0:
        raise ValueError("q and variance must be finite and positive")
    root = np.sqrt(q)
    return variance * (1 - root) ** 2, variance * (1 + root) ** 2


def mp_density(values, q, variance=1.0):
    """Continuous MP density; for q > 1 a separate atom at zero has mass 1-1/q."""
    lo, hi = mp_support(q, variance)
    x = np.asarray(values, dtype=float)
    result = np.zeros_like(x)
    inside = (x > lo) & (x < hi) & (x > 0)
    result[inside] = np.sqrt((hi - x[inside]) * (x[inside] - lo)) / (
        2 * np.pi * q * variance * x[inside]
    )
    return result


def rmt_clip(x):
    """Replace correlation eigenvalues below the unit-noise MP edge by their mean.

    The edge is a null-model reference, not a fitted or oracle threshold.
    Re-normalization restores unit correlation diagonal after reconstruction.
    """
    x = validate_panel(x)
    sample = np.cov(x, rowvar=False, ddof=1)
    sd = np.sqrt(np.diag(sample))
    corr = sample / np.outer(sd, sd)
    eig, vec = np.linalg.eigh((corr + corr.T) / 2)
    eig = np.maximum(eig, 0)
    edge = mp_support(x.shape[1] / (x.shape[0] - 1))[1]
    bulk = eig <= edge
    cleaned = eig.copy()
    if np.any(bulk):
        cleaned[bulk] = eig[bulk].mean()
    rebuilt = (vec * cleaned) @ vec.T
    scale = np.sqrt(np.diag(rebuilt))
    rebuilt = rebuilt / np.outer(scale, scale)
    covariance = rebuilt * np.outer(sd, sd)
    return (covariance + covariance.T) / 2


METHODS = ("sample", "diagonal", "ledoit_wolf", "rmt_clip")


def estimate(x, method):
    x = validate_panel(x)
    if method == "rmt_clip":
        return rmt_clip(x)
    if method == "ledoit_wolf":
        # Match the other estimators' n-1 covariance normalization.
        return LedoitWolf().fit(x).covariance_ * len(x) / (len(x) - 1)
    sample = np.cov(x, rowvar=False, ddof=1)
    if method == "sample":
        return sample
    if method == "diagonal":
        return np.diag(np.diag(sample))
    raise ValueError(f"Unknown covariance method: {method}")

import numpy as np
import pytest
from scipy.integrate import quad
from sklearn.covariance import LedoitWolf

from covariance_shift.estimators import METHODS, estimate, mp_density, mp_support


@pytest.mark.parametrize("q", [0.25, 1.0, 2.0])
def test_mp_continuous_mass(q):
    lo, hi = mp_support(q)
    mass = quad(lambda t: float(mp_density(t, q)), lo, hi)[0]
    assert mass == pytest.approx(min(1, 1 / q), abs=1e-7)


@pytest.mark.parametrize("q", [0, -1, float("nan")])
def test_bad_mp_ratio(q):
    with pytest.raises(ValueError):
        mp_support(q)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("shape", [(12, 20), (40, 10)])
def test_covariance_invariants(method, shape):
    x = np.random.default_rng(3).normal(size=shape)
    c = estimate(x, method)
    assert c.shape == (shape[1], shape[1])
    assert np.allclose(c, c.T)
    assert np.linalg.eigvalsh(c).min() > -1e-10
    if method != "ledoit_wolf":
        assert np.allclose(np.diag(c), x.var(axis=0, ddof=1))


@pytest.mark.parametrize("method", METHODS)
def test_translation_and_scalar_equivariance(method):
    x = np.random.default_rng(8).normal(size=(60, 8))
    assert np.allclose(estimate(2 * x + 3, method), 4 * estimate(x, method))


def test_lw_normalization():
    x = np.random.default_rng(18).normal(size=(30, 8))
    assert np.allclose(estimate(x, "ledoit_wolf"), LedoitWolf().fit(x).covariance_ * 30 / 29)


@pytest.mark.parametrize("bad", [np.ones((5, 3)), [[1, 2]], [[1, 2], [2, np.nan]]])
def test_reject_invalid_panels(bad):
    with pytest.raises(ValueError):
        estimate(bad, "rmt_clip")


def test_rmt_is_diagonal_if_no_eigenvalue_exceeds_edge():
    x = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=float)
    assert np.allclose(estimate(x, "rmt_clip"), np.eye(2) * 4 / 3)


def test_rmt_preserves_a_strong_common_factor():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(300, 1)) + 0.2 * rng.normal(size=(300, 10))
    c = estimate(x, "rmt_clip")
    corr = c / np.sqrt(np.outer(np.diag(c), np.diag(c)))
    assert corr[0, 1] > 0.8

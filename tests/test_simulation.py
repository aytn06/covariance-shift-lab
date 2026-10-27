import numpy as np
import pytest

from covariance_shift.experiment import monte_carlo, paired_comparisons, summarize, write_csv
from covariance_shift.simulation import SCENARIOS, draw_returns, generate_path, generate_sample


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_scenario_reproducibility(scenario):
    a = generate_sample(scenario, 8, 20, 123)
    b = generate_sample(scenario, 8, 20, 123)
    assert np.array_equal(a.train, b.train)
    assert a.train.shape == (20, 8)
    assert np.linalg.eigvalsh(a.future_covariance).min() > 0


def test_t_noise_has_the_stated_covariance():
    truth = np.array([[1, 0.3], [0.3, 2]])
    draws = draw_returns(truth, 200_000, np.random.default_rng(45), df=5)
    assert np.linalg.norm(np.cov(draws, rowvar=False) - truth) / np.linalg.norm(truth) < 0.03


def test_ground_truth_changes_at_specified_time():
    _, truth = generate_path(n_assets=5, days=70, change_day=40, seed=6)
    assert np.array_equal(truth[0], truth[39])
    assert np.array_equal(truth[40], truth[-1])
    assert not np.allclose(truth[39], truth[40])


def test_small_end_to_end_study():
    rows = monte_carlo(5, [8], 3, 99)
    summary = summarize(rows)
    pairs = paired_comparisons(rows, 99)
    assert len(rows) == 5 * 3 * 4
    assert len(summary) == 20
    assert len(pairs) == 20
    assert all(row["oracle_risk_ratio"] >= 1 - 1e-5 for row in rows)
    assert all(np.isfinite(row["true_to_forecast_variance"]) for row in rows)
    assert paired_comparisons(rows, 99) == pairs


def test_csv_output_uses_portable_lf_lines(tmp_path):
    output = tmp_path / "result.csv"
    write_csv(output, [{"trial": 1, "value": 2.5}])
    assert output.read_bytes() == b"trial,value\n1,2.5\n"

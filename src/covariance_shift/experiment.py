"""Reproducible synthetic study, figures, raw evidence, and paired comparisons."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np

from .estimators import METHODS, estimate, mp_density, mp_support
from .portfolio import minimum_variance, walk_forward
from .simulation import SCENARIOS, generate_path, generate_sample

METRICS = ("covariance_error", "oracle_risk_ratio", "true_to_forecast_variance")


def write_csv(path, rows):
    if not rows:
        raise ValueError("No results to write")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def monte_carlo(n_assets, windows, repeats, seed):
    rows = []
    for s, scenario in enumerate(SCENARIOS):
        for observations in windows:
            for trial in range(repeats):
                key = [seed, s, observations, trial]
                sample = generate_sample(scenario, n_assets, observations, key)
                truth = sample.future_covariance
                oracle = minimum_variance(truth)
                oracle_variance = oracle @ truth @ oracle
                for method in METHODS:
                    covariance = estimate(sample.train, method)
                    weights = minimum_variance(covariance)
                    true_variance = weights @ truth @ weights
                    rows.append({
                        "scenario": scenario, "observations": observations,
                        "assets": n_assets, "trial": trial, "method": method,
                        "covariance_error": float(np.linalg.norm(covariance - truth, "fro") / np.linalg.norm(truth, "fro")),
                        "oracle_risk_ratio": float(true_variance / oracle_variance),
                        "true_to_forecast_variance": float(true_variance / (weights @ covariance @ weights)),
                        "max_weight": float(weights.max()),
                        "equal_weight_risk_ratio": float(np.mean(truth) / oracle_variance),
                    })
            print(f"Completed {scenario}: {observations} observations, {repeats} paired trials", flush=True)
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["scenario"], row["observations"], row["method"]), []).append(row)
    summary = []
    for (scenario, observations, method), group in groups.items():
        record = {"scenario": scenario, "observations": observations, "method": method, "trials": len(group)}
        for metric in METRICS:
            values = np.array([r[metric] for r in group])
            record[metric + "_mean"] = float(values.mean())
            record[metric + "_se"] = float(values.std(ddof=1) / np.sqrt(len(values)))
        summary.append(record)
    return summary


def paired_comparisons(rows, seed):
    groups = {}
    for row in rows:
        key = (row["scenario"], row["observations"])
        groups.setdefault(key, {}).setdefault(row["trial"], {})[row["method"]] = row
    rng = np.random.default_rng([seed, 9001])
    comparisons = []
    for (scenario, observations), trials in groups.items():
        for baseline in ("sample", "ledoit_wolf"):
            for metric in ("covariance_error", "oracle_risk_ratio"):
                delta = np.array([r["rmt_clip"][metric] - r[baseline][metric] for r in trials.values()])
                boot = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
                lo, hi = np.quantile(boot, [0.025, 0.975])
                comparisons.append({"scenario": scenario, "observations": observations,
                                    "comparison": "rmt_clip_minus_" + baseline, "metric": metric,
                                    "mean_difference": float(delta.mean()),
                                    "bootstrap_low": float(lo), "bootstrap_high": float(hi)})
    return comparisons


def rolling_study(seed):
    returns, truth = generate_path(seed=seed)
    daily_rows, risk_rows, summaries = [], [], []
    for method in (*METHODS, "equal_weight"):
        for cost_bps in (0, 5, 20):
            daily, allocations = walk_forward(returns, method, cost_bps=cost_bps)
            daily_rows.extend({"method": method, "cost_bps": cost_bps, **row} for row in daily)
            net = np.array([r["net_return"] for r in daily])
            nav = np.cumprod(1 + net)
            peaks = np.maximum.accumulate(np.r_[1.0, nav])[1:]
            summaries.append({"method": method, "cost_bps": cost_bps,
                              "annualized_net_volatility": float(net.std(ddof=1) * np.sqrt(252)),
                              "total_net_return": float(nav[-1] - 1),
                              "max_drawdown": float(np.min(nav / peaks - 1)),
                              "total_traded_notional": float(sum(r["turnover"] for r in daily))})
            if cost_bps == 0:
                for allocation in allocations:
                    day, weights = allocation["day"], allocation["weights"]
                    actual = float(weights @ truth[day] @ weights)
                    risk_rows.append({"method": method, "day": day,
                                      "forecast_variance": allocation["forecast_variance"],
                                      "true_variance": actual,
                                      "true_to_forecast_variance": actual / allocation["forecast_variance"],
                                      "turnover": allocation["turnover"]})
    return daily_rows, risk_rows, summaries


def figures(out, rows, risk_rows, seed, reference_window):
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 160})
    colors = {"sample": "#687784", "diagonal": "#bd8332", "ledoit_wolf": "#126d67", "rmt_clip": "#bd493c", "equal_weight": "#816e9c"}
    labels = {"sample": "Sample", "diagonal": "Diagonal", "ledoit_wolf": "Ledoit-Wolf", "rmt_clip": "RMT clip", "equal_weight": "Equal weight"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.7))
    for ax, metric, title in zip(axes, METRICS[:2], ("Covariance error (lower is better)", "Variance / oracle variance (best = 1)")):
        for j, method in enumerate(METHODS):
            group = [next(r for r in rows if r["scenario"] == s and r["observations"] == reference_window and r["method"] == method) for s in SCENARIOS]
            pos = np.arange(len(SCENARIOS)) + (j - 1.5) * 0.17
            ax.errorbar(pos, [r[metric + "_mean"] for r in group], yerr=[1.96 * r[metric + "_se"] for r in group],
                        fmt="o", ms=4, capsize=3, color=colors[method], label=labels[method])
        ax.set_xticks(range(len(SCENARIOS)), [s.replace("_", "\n") for s in SCENARIOS])
        ax.set_title(title, loc="left", fontsize=12)
        ax.grid(axis="y", alpha=0.18)
    axes[1].axhline(1, color="#bbbbbb", lw=1, ls="--")
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    fig.suptitle(f"Synthetic covariance stress test | {reference_window} training observations", x=0.06, ha="left", fontweight="bold")
    fig.text(0.06, 0.015, "Points: Monte Carlo means. Bars: mean +/- 1.96 standard errors, not simultaneous intervals.", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(out / "estimator_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for method in METHODS:
        group = [r for r in risk_rows if r["method"] == method]
        ax.plot([r["day"] for r in group], [r["true_to_forecast_variance"] for r in group], label=labels[method], color=colors[method], lw=1.7)
    ax.axvline(480, color="#333333", ls=":", label="Covariance changes")
    ax.axhline(1, color="#999999", lw=1, ls="--")
    ax.set(xlabel="Synthetic trading day", ylabel="True variance / forecast variance", title="A better estimate of yesterday can still miss today's risk")
    ax.legend(frameon=False, fontsize=9, ncol=3)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out / "regime_change.png")
    plt.close(fig)

    rng = np.random.default_rng([seed, 73])
    panel = rng.normal(size=(256, 128))
    eig = np.linalg.eigvalsh(np.cov(panel, rowvar=False))
    q = 128 / 255
    lo, hi = mp_support(q)
    grid = np.linspace(lo, hi, 600)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(eig, bins=25, density=True, color="#126d67", alpha=0.5, label="One finite Gaussian sample")
    ax.plot(grid, mp_density(grid, q), color="#bd493c", lw=2, label="Asymptotic MP density")
    ax.set(xlabel="Sample covariance eigenvalue", ylabel="Density", title="Null-model reference, not a finite-sample guarantee")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "mp_reference.png")
    plt.close(fig)


def report(out, summary, paired, rolling, repeats, windows, n_assets):
    lines = ["# Computational Results", "", "All results below use synthetic, zero-mean returns. No market data or trading alpha is claimed.", "",
             f"Design: {n_assets} assets; training windows {windows}; {repeats} independent trials per scenario/window. Methods share the same data in each trial.", "",
             "## Covariance and allocation", "", "Ratios use variance, not volatility. Oracle risk uses the known future covariance with the same long-only constraints. Forecast ratios above 1 mean understated risk.", "",
             "| Scenario | Observations | Method | Relative covariance error | Variance / oracle | True / forecast variance |",
             "|---|---:|---|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['scenario']} | {r['observations']} | {r['method']} | {r['covariance_error_mean']:.3f} | {r['oracle_risk_ratio_mean']:.3f} | {r['true_to_forecast_variance_mean']:.3f} |")
    lines += ["", "## Paired evidence", "", "Differences below are RMT minus the baseline in variance/oracle variance. Negative favors RMT; positive favors the baseline. These are exploratory percentile bootstrap intervals across independent trials, with no multiple-comparison correction.", "",
              "| Scenario | Observations | Comparison | Mean difference | Bootstrap 95% interval |", "|---|---:|---|---:|---|"]
    for r in paired:
        if r["metric"] == "oracle_risk_ratio":
            lines.append(f"| {r['scenario']} | {r['observations']} | {r['comparison']} | {r['mean_difference']:.3f} | [{r['bootstrap_low']:.3f}, {r['bootstrap_high']:.3f}] |")
    lines += ["", "## Rolling cost sensitivity", "", "One 840-day Gaussian path, 24 assets, 96-day lookback, 21-day rebalancing, covariance break on day 480. Holdings drift between trades. Costs apply to total bought plus sold notional, including entry from cash. This single path is an accounting and shift-response demonstration, not a performance claim.", "",
              "| Method | Cost (bps per traded dollar) | Net annualized volatility | Total net return | Max drawdown | Total traded NAV |", "|---|---:|---:|---:|---:|---:|"]
    for r in rolling:
        lines.append(f"| {r['method']} | {r['cost_bps']} | {r['annualized_net_volatility']:.2%} | {r['total_net_return']:.2%} | {r['max_drawdown']:.2%} | {r['total_traded_notional']:.2f} |")
    lines += ["", "## Limits", "", "The MP edge assumes an independent, isotropic noise reference. Real factor structure, heavy tails, serial dependence, and changing covariance do not satisfy that reference. Clipping also preserves noisy sample variances. The scenario grid is fixed and small, and the rolling path has only one seed. No universal ranking, new theorem, or production readiness is established.", ""]
    (out / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--repeats", type=int, default=60)
    parser.add_argument("--assets", type=int, default=32)
    parser.add_argument("--windows", type=int, nargs="+", default=[24, 64, 160])
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if args.repeats < 2 or args.assets < 2 or min(args.windows) < 4 or len(set(args.windows)) != len(args.windows) or args.seed < 0:
        parser.error("Use at least 2 trials/assets, unique windows >= 4, and a nonnegative seed")
    args.output.mkdir(parents=True, exist_ok=True)
    raw = monte_carlo(args.assets, args.windows, args.repeats, args.seed)
    summary = summarize(raw)
    paired = paired_comparisons(raw, args.seed)
    daily, risks, rolling = rolling_study(args.seed + 1)
    for name, rows in (("trials", raw), ("estimator_summary", summary), ("paired_comparisons", paired),
                       ("rolling_daily", daily), ("rolling_risk", risks), ("rolling_summary", rolling)):
        write_csv(args.output / f"{name}.csv", rows)
    figures(args.output, summary, risks, args.seed, args.windows[len(args.windows) // 2])
    report(args.output, summary, paired, rolling, args.repeats, args.windows, args.assets)
    source = Path(__file__).parent
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))}
    manifest = {"data": "synthetic only", "seed": args.seed, "repeats": args.repeats,
                "assets": args.assets, "windows": args.windows, "scenarios": SCENARIOS,
                "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "scikit-learn", "matplotlib")},
                "source_sha256": hashes}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote evidence and figures to {args.output}")


if __name__ == "__main__":
    main()

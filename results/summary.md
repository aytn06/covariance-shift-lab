# Computational Results

All results below use synthetic, zero-mean returns. No market data or trading alpha is claimed.

Design: 32 assets; training windows [24, 64, 160]; 60 independent trials per scenario/window. Methods share the same data in each trial.

## Covariance and allocation

Ratios use variance, not volatility. Oracle risk uses the known future covariance with the same long-only constraints. Forecast ratios above 1 mean understated risk.

| Scenario | Observations | Method | Relative covariance error | Variance / oracle | True / forecast variance |
|---|---:|---|---:|---:|---:|
| independent | 24 | sample | 1.077 | 2.558 | 57.266 |
| independent | 24 | diagonal | 0.291 | 1.097 | 1.209 |
| independent | 24 | ledoit_wolf | 0.425 | 1.317 | 1.030 |
| independent | 24 | rmt_clip | 0.291 | 1.097 | 1.209 |
| independent | 64 | sample | 0.643 | 1.569 | 2.861 |
| independent | 64 | diagonal | 0.176 | 1.030 | 1.063 |
| independent | 64 | ledoit_wolf | 0.376 | 1.255 | 0.992 |
| independent | 64 | rmt_clip | 0.180 | 1.031 | 1.066 |
| independent | 160 | sample | 0.404 | 1.207 | 1.479 |
| independent | 160 | diagonal | 0.110 | 1.012 | 1.029 |
| independent | 160 | ledoit_wolf | 0.302 | 1.159 | 0.955 |
| independent | 160 | rmt_clip | 0.114 | 1.013 | 1.030 |
| weak_factors | 24 | sample | 0.989 | 1.932 | 8.184 |
| weak_factors | 24 | diagonal | 0.468 | 1.277 | 3.428 |
| weak_factors | 24 | ledoit_wolf | 0.503 | 1.512 | 2.104 |
| weak_factors | 24 | rmt_clip | 0.513 | 1.301 | 2.828 |
| weak_factors | 64 | sample | 0.601 | 1.390 | 2.269 |
| weak_factors | 64 | diagonal | 0.416 | 1.242 | 3.143 |
| weak_factors | 64 | ledoit_wolf | 0.419 | 1.339 | 1.629 |
| weak_factors | 64 | rmt_clip | 0.342 | 1.190 | 1.609 |
| weak_factors | 160 | sample | 0.373 | 1.144 | 1.330 |
| weak_factors | 160 | diagonal | 0.397 | 1.238 | 3.070 |
| weak_factors | 160 | ledoit_wolf | 0.314 | 1.176 | 1.258 |
| weak_factors | 160 | rmt_clip | 0.235 | 1.096 | 1.189 |
| factors | 24 | sample | 0.548 | 1.386 | 3.073 |
| factors | 24 | diagonal | 0.877 | 1.845 | 11.084 |
| factors | 24 | ledoit_wolf | 0.499 | 1.393 | 2.160 |
| factors | 24 | rmt_clip | 0.499 | 1.385 | 2.365 |
| factors | 64 | sample | 0.361 | 1.147 | 1.363 |
| factors | 64 | diagonal | 0.863 | 1.880 | 9.649 |
| factors | 64 | ledoit_wolf | 0.335 | 1.159 | 1.283 |
| factors | 64 | rmt_clip | 0.337 | 1.133 | 1.249 |
| factors | 160 | sample | 0.213 | 1.054 | 1.119 |
| factors | 160 | diagonal | 0.866 | 1.898 | 9.897 |
| factors | 160 | ledoit_wolf | 0.210 | 1.058 | 1.087 |
| factors | 160 | rmt_clip | 0.201 | 1.053 | 1.088 |
| heavy_tails | 24 | sample | 0.752 | 1.585 | 3.722 |
| heavy_tails | 24 | diagonal | 0.895 | 1.925 | 11.963 |
| heavy_tails | 24 | ledoit_wolf | 0.606 | 1.611 | 2.593 |
| heavy_tails | 24 | rmt_clip | 0.675 | 1.585 | 3.289 |
| heavy_tails | 64 | sample | 0.575 | 1.221 | 1.648 |
| heavy_tails | 64 | diagonal | 0.884 | 1.884 | 10.143 |
| heavy_tails | 64 | ledoit_wolf | 0.462 | 1.282 | 1.509 |
| heavy_tails | 64 | rmt_clip | 0.523 | 1.241 | 1.567 |
| heavy_tails | 160 | sample | 0.335 | 1.100 | 1.225 |
| heavy_tails | 160 | diagonal | 0.871 | 1.895 | 9.874 |
| heavy_tails | 160 | ledoit_wolf | 0.304 | 1.117 | 1.180 |
| heavy_tails | 160 | rmt_clip | 0.307 | 1.093 | 1.171 |
| regime_shift | 24 | sample | 0.600 | 2.197 | 3.722 |
| regime_shift | 24 | diagonal | 0.920 | 2.491 | 16.693 |
| regime_shift | 24 | ledoit_wolf | 0.628 | 2.145 | 3.250 |
| regime_shift | 24 | rmt_clip | 0.587 | 2.256 | 3.205 |
| regime_shift | 64 | sample | 0.526 | 1.759 | 2.057 |
| regime_shift | 64 | diagonal | 0.917 | 2.439 | 15.708 |
| regime_shift | 64 | ledoit_wolf | 0.559 | 1.774 | 2.060 |
| regime_shift | 64 | rmt_clip | 0.536 | 1.774 | 1.903 |
| regime_shift | 160 | sample | 0.504 | 1.646 | 1.721 |
| regime_shift | 160 | diagonal | 0.918 | 2.461 | 16.061 |
| regime_shift | 160 | ledoit_wolf | 0.522 | 1.649 | 1.728 |
| regime_shift | 160 | rmt_clip | 0.519 | 1.685 | 1.692 |

## Paired evidence

Differences below are RMT minus the baseline in variance/oracle variance. Negative favors RMT; positive favors the baseline. These are exploratory percentile bootstrap intervals across independent trials, with no multiple-comparison correction.

| Scenario | Observations | Comparison | Mean difference | Bootstrap 95% interval |
|---|---:|---|---:|---|
| independent | 24 | rmt_clip_minus_sample | -1.461 | [-1.579, -1.355] |
| independent | 24 | rmt_clip_minus_ledoit_wolf | -0.219 | [-0.240, -0.200] |
| independent | 64 | rmt_clip_minus_sample | -0.538 | [-0.572, -0.503] |
| independent | 64 | rmt_clip_minus_ledoit_wolf | -0.224 | [-0.238, -0.209] |
| independent | 160 | rmt_clip_minus_sample | -0.194 | [-0.207, -0.182] |
| independent | 160 | rmt_clip_minus_ledoit_wolf | -0.146 | [-0.156, -0.137] |
| weak_factors | 24 | rmt_clip_minus_sample | -0.631 | [-0.702, -0.561] |
| weak_factors | 24 | rmt_clip_minus_ledoit_wolf | -0.211 | [-0.248, -0.174] |
| weak_factors | 64 | rmt_clip_minus_sample | -0.200 | [-0.228, -0.172] |
| weak_factors | 64 | rmt_clip_minus_ledoit_wolf | -0.149 | [-0.172, -0.127] |
| weak_factors | 160 | rmt_clip_minus_sample | -0.049 | [-0.058, -0.039] |
| weak_factors | 160 | rmt_clip_minus_ledoit_wolf | -0.080 | [-0.090, -0.069] |
| factors | 24 | rmt_clip_minus_sample | -0.001 | [-0.054, 0.053] |
| factors | 24 | rmt_clip_minus_ledoit_wolf | -0.008 | [-0.058, 0.050] |
| factors | 64 | rmt_clip_minus_sample | -0.014 | [-0.029, 0.002] |
| factors | 64 | rmt_clip_minus_ledoit_wolf | -0.026 | [-0.044, -0.006] |
| factors | 160 | rmt_clip_minus_sample | -0.001 | [-0.008, 0.004] |
| factors | 160 | rmt_clip_minus_ledoit_wolf | -0.005 | [-0.012, 0.001] |
| heavy_tails | 24 | rmt_clip_minus_sample | 0.001 | [-0.074, 0.075] |
| heavy_tails | 24 | rmt_clip_minus_ledoit_wolf | -0.026 | [-0.088, 0.036] |
| heavy_tails | 64 | rmt_clip_minus_sample | 0.019 | [-0.006, 0.047] |
| heavy_tails | 64 | rmt_clip_minus_ledoit_wolf | -0.041 | [-0.079, -0.007] |
| heavy_tails | 160 | rmt_clip_minus_sample | -0.007 | [-0.016, 0.002] |
| heavy_tails | 160 | rmt_clip_minus_ledoit_wolf | -0.024 | [-0.035, -0.013] |
| regime_shift | 24 | rmt_clip_minus_sample | 0.058 | [-0.049, 0.172] |
| regime_shift | 24 | rmt_clip_minus_ledoit_wolf | 0.111 | [0.011, 0.223] |
| regime_shift | 64 | rmt_clip_minus_sample | 0.015 | [-0.032, 0.064] |
| regime_shift | 64 | rmt_clip_minus_ledoit_wolf | -0.000 | [-0.047, 0.051] |
| regime_shift | 160 | rmt_clip_minus_sample | 0.040 | [0.011, 0.068] |
| regime_shift | 160 | rmt_clip_minus_ledoit_wolf | 0.036 | [0.008, 0.066] |

## Rolling cost sensitivity

One 840-day Gaussian path, 24 assets, 96-day lookback, 21-day rebalancing, covariance break on day 480. Holdings drift between trades. Costs apply to total bought plus sold notional, including entry from cash. This single path is an accounting and shift-response demonstration, not a performance claim.

| Method | Cost (bps per traded dollar) | Net annualized volatility | Total net return | Max drawdown | Total traded NAV |
|---|---:|---:|---:|---:|---:|
| sample | 0 | 11.35% | -30.92% | -38.26% | 13.85 |
| sample | 5 | 11.35% | -31.40% | -38.60% | 13.85 |
| sample | 20 | 11.35% | -32.81% | -39.59% | 13.85 |
| diagonal | 0 | 13.73% | -38.35% | -46.92% | 4.35 |
| diagonal | 5 | 13.73% | -38.48% | -46.99% | 4.35 |
| diagonal | 20 | 13.74% | -38.88% | -47.18% | 4.35 |
| ledoit_wolf | 0 | 11.42% | -33.60% | -39.91% | 12.87 |
| ledoit_wolf | 5 | 11.42% | -34.02% | -40.21% | 12.87 |
| ledoit_wolf | 20 | 11.43% | -35.29% | -41.10% | 12.87 |
| rmt_clip | 0 | 11.48% | -31.50% | -39.09% | 13.66 |
| rmt_clip | 5 | 11.48% | -31.97% | -39.42% | 13.66 |
| rmt_clip | 20 | 11.48% | -33.35% | -40.40% | 13.66 |
| equal_weight | 0 | 15.52% | -40.65% | -50.14% | 2.59 |
| equal_weight | 5 | 15.52% | -40.73% | -50.16% | 2.59 |
| equal_weight | 20 | 15.52% | -40.96% | -50.24% | 2.59 |

## Limits

The MP edge assumes an independent, isotropic noise reference. Real factor structure, heavy tails, serial dependence, and changing covariance do not satisfy that reference. Clipping also preserves noisy sample variances. The scenario grid is fixed and small, and the rolling path has only one seed. No universal ranking, new theorem, or production readiness is established.

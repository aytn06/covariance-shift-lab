# Definitions and Assumptions

## Data and covariance

Rows of X are observations and columns are assets. Each estimator centers each
column. Sample covariance uses denominator T-1. The scikit-learn Ledoit-Wolf
estimate uses denominator T internally, so its final matrix is multiplied by
T/(T-1) to match the reporting convention here. This does not change its weights.

Synthetic factor covariances are formed from B B^T plus a positive diagonal,
normalized to correlation, then scaled by heterogeneous asset volatilities.
Independent assets use the diagonal of this covariance. The weak-factor setting
scales B B^T by 0.12. Heavy tails use a multivariate t distribution with five
degrees of freedom and covariance equal to the stated ground truth.

In the shift scenario the first two-thirds of the training window come from an
old covariance and the remaining observations from a new one. Evaluation uses
the new covariance. Both volatility and factor loadings change. This is a
controlled stale-data test, not a model of any specific market episode.

## RMT clipping

Let q = N/(T-1). The unit-noise MP reference support is
[(1-sqrt(q))^2, (1+sqrt(q))^2]. Its continuous density is
sqrt((upper-x)(x-lower))/(2*pi*q*x); when q>1 it has an additional point mass
1-1/q at zero. The function mp_density returns only the continuous part.

The algorithm keeps eigenvalues above the upper edge and replaces the remaining
ones, including numerical zeros, by their average. After reconstruction, a
diagonal congruence restores unit diagonal. This last step can change the
eigenvalues again. Finally the original sample standard deviations restore the
covariance scale. This is a heuristic motivated by the MP null reference, not a
claim that financial correlation eigenvalues obey the null exactly.

## Allocation and forecasts

All optimized portfolios minimize w^T C w subject to w_i >= 0 and sum(w)=1.
The solver divides C by its average diagonal, adds 1e-8 times the identity for
numerical stability, and checks a simplex optimality gap. The same procedure
applies to the oracle. Forecasts and evaluation use the original covariance,
not the numerical ridge. No expected return, leverage, or short selling is used.

At trading day d, the estimate uses returns [d-lookback, d), never return d.
The target is applied before day d's return. Between trades, positions drift:
w_i,new = w_i,old*(1+r_i)/(1+w^T r). Turnover is sum(abs(target-drifted holdings)).
The initial trade begins in cash and has turnover 1. Costs are c*turnover where
c is basis points/10000, and net growth is (1-cost)*(1+gross return).

This is a proportional-cost convention charged against pre-trade NAV; it does
not solve an exact cash-and-trade-cost fixed point or model market impact.
The rolling forecast comparison uses true variance on the rebalance day, not
average future holding-period variance. Equal weight uses sample covariance
only to supply a forecast; its weights do not depend on any estimate.

## Evidence status

Positive semidefiniteness of spectral averaging and positive diagonal
congruence is a linear-algebra fact. MP limits require their own asymptotic
assumptions. Method rankings and risk-calibration figures in this repository
are computational observations only. No new theorem is proposed.

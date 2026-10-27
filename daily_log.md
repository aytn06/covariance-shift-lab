# Research Log

## October 26, 2026

Started a separate implementation of covariance-estimation stress tests, rather
than modifying or relabeling another repository. Defined the scenario grid,
methods, long-only allocation constraints, and evaluation rules before running
the first experiment. Implemented and tested the synthetic generators,
conservative MP clipping, shrinkage baseline, risk metrics, and rolling costs.

The first run completed all planned experiments; 43 tests passed after adding a
CSV line-ending regression check. Retained
negative results and the distinction between covariance accuracy, allocation
quality, and forecast calibration. Known limitations are recorded in the README
and definitions. No novelty or real-market performance claim is made.

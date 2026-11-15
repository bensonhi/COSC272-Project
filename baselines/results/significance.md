# Seed ensembles and paired bootstrap

| family | seeds | macro-F1 (mean / ensemble) | ECE (mean / ensemble) | selectivity | evidence_flip pair | evidence_weaken pair |
|---|---|---|---|---|---|---|
| B2 original-data SFT | 3 | 0.855 / 0.865 | 0.119 / 0.080 | +0.329 / +0.343 | 0.784 / 0.797 | 0.190 / 0.214 |
| augmented SFT | 2 | 0.843 / 0.852 | 0.127 / 0.100 | +0.638 / +0.647 | 0.672 / 0.727 | 0.286 / 0.214 |
| + all counterfactuals | 2 | 0.835 / 0.835 | 0.139 / 0.119 | +0.640 / +0.633 | 0.863 / 0.875 | 0.643 / 0.643 |
| + evidence-side counterfactuals | 3 | 0.841 / 0.849 | 0.134 / 0.101 | +0.641 / +0.652 | 0.844 / 0.875 | 0.738 / 0.786 |

Difference of seed-averaged metrics, 95% CI from 1000 paired bootstrap resamples of the test claims.

| comparison | macro_f1 | selectivity | flip_pair | weaken_pair |
|---|---|---|---|---|
| + evidence-side counterfactuals vs augmented SFT | -0.003 [-0.027, +0.020] | +0.003 [-0.031, +0.041] | +0.172 [+0.117, +0.237] | +0.452 [+0.167, +0.726] |
| + evidence-side counterfactuals vs B2 original-data SFT | -0.015 [-0.038, +0.009] | +0.312 [+0.243, +0.385] | +0.060 [+0.014, +0.110] | +0.548 [+0.311, +0.810] |
| augmented SFT vs B2 original-data SFT | -0.012 [-0.036, +0.013] | +0.309 [+0.241, +0.381] | -0.112 [-0.163, -0.066] | +0.095 [-0.042, +0.250] |
| + all counterfactuals vs augmented SFT | -0.009 [-0.036, +0.018] | +0.002 [-0.039, +0.044] | +0.191 [+0.127, +0.260] | +0.357 [+0.156, +0.571] |

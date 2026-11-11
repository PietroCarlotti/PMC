# Temporary sparse GP VI benchmarks

This is a provisional snapshot of the Gaussian K-scaling experiments. It does not change the official GP implementation or the PMC_AISTATS results.

- `results_vi_gp_k*_s*.csv` records ten variational FITC GP fits per setting, with ten K-means inducing points. The temporary Pyro implementation uses a joint full-covariance variational Gaussian over whitened inducing values and transformed hyperparameters, 1,000 Adam steps at learning rate 0.01, four ELBO particles, batches of up to 1,024 pilots, and 2,000 posterior predictive draws. These runs generated float64 data.
- `mse_convergence.png` and `runtime_convergence.png` overlay those GP results on the saved GBF, PMC-MLP, and PMC-OLS results in PMC_AISTATS. **The plotted methods did not use matched draws.** The saved MLP and OLS runs generated float32 data; their original device and raw draws were not recorded. Treat the curves as provisional method comparisons.
- `paired_k1000_s10_results.csv` is a fresh ten-replicate CPU comparison. For each replicate, GP, MLP, and OLS read the same saved float32 sample tensor. GP converted that tensor to float64 for its computation. The `data_sha256` column verifies the shared input within each replicate. Its results are separate from the convergence plots.

Runtimes exclude data generation. The paired CPU runtimes and saved baseline runtimes come from different runs and should not be compared as matched timings.

// Data
data {
  int<lower=1> S;
  int<lower=1> m;
  int<lower=1> r;
  matrix[S, m] D;
  matrix[S, r] C;
  vector[m] D_zero;
  real<lower=0> length_scale_prior_shape;
  real<lower=0> length_scale_prior_scale;
  real<lower=0> signal_sd_prior_scale;
  real<lower=0> noise_sd_prior_scale;
}

// Transformed data
transformed data {
  array[S] vector[m] D_vectors;
  array[1] vector[m] D_zero_vectors;

  for (s in 1:S) {
    D_vectors[s] = D[s]';
  }
  D_zero_vectors[1] = D_zero;
}

// Parameters
parameters {
  real<lower=0> length_scale;
  vector<lower=0>[r] signal_sd;
  vector<lower=0>[r] noise_sd;
}

// Model
model {
  // Set the hyperparameter priors
  length_scale ~ inv_gamma(length_scale_prior_shape, length_scale_prior_scale);
  signal_sd ~ normal(0, signal_sd_prior_scale);
  noise_sd ~ normal(0, noise_sd_prior_scale);

  // Fit a Gaussian process for each coordinate of the target
  for (j in 1:r) {
    // Build the latent function covariance matrix
    matrix[S, S] covariance = gp_exp_quad_cov(D_vectors, signal_sd[j], length_scale);

    // Add independent observation noise
    covariance = add_diag(covariance, square(noise_sd[j]));

    // Model the observed target coordinate
    C[, j] ~ multi_normal_cholesky(
      rep_vector(0, S),
      cholesky_decompose(covariance)
    );
  }
}

// Generated quantities
generated quantities {
  // Store one posterior draw of the latent target at zero discrepancy
  vector[r] f_zero;

  // Draw the latent function at zero discrepancy for each target coordinate
  for (j in 1:r) {
    // Build the latent function covariance matrix
    matrix[S, S] covariance = gp_exp_quad_cov(D_vectors, signal_sd[j], length_scale);

    // Build the training-to-zero cross-covariance
    matrix[S, 1] cross_covariance = gp_exp_quad_cov(
      D_vectors,
      D_zero_vectors,
      signal_sd[j],
      length_scale
    );

    // Add independent observation noise
    covariance = add_diag(covariance, square(noise_sd[j]));

    // Declare the conditional mean and variance at zero discrepancy
    real f_zero_mean;
    real f_zero_variance;

    // Compute the conditional mean and variance at zero discrepancy
    f_zero_mean = dot_product(
      cross_covariance[, 1],
      mdivide_left_spd(covariance, C[, j])
    );
    f_zero_variance = square(signal_sd[j]) - dot_product(
      cross_covariance[, 1],
      mdivide_left_spd(covariance, cross_covariance[, 1])
    );

    // Draw the latent function at zero discrepancy
    f_zero[j] = normal_rng(
      f_zero_mean,
      sqrt(fmax(f_zero_variance, 0))
    );
  }
}

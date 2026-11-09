// Data
data {
  int<lower=1> S;
  int<lower=1> m;
  int<lower=1> r;
  int<lower=1> num_inducing;
  matrix[S, m] D;
  matrix[S, r] C;
  matrix[num_inducing, m] Z;
  vector[m] D_zero;
  real<lower=0> length_scale_prior_shape;
  real<lower=0> length_scale_prior_scale;
  real<lower=0> signal_sd_prior_scale;
  real<lower=0> noise_sd_prior_scale;
}

// Transformed data
transformed data {
  array[S] vector[m] D_vectors;
  array[num_inducing] vector[m] Z_vectors;
  array[1] vector[m] D_zero_vectors;

  for (s in 1:S) {
    D_vectors[s] = D[s]';
  }
  for (i in 1:num_inducing) {
    Z_vectors[i] = Z[i]';
  }
  D_zero_vectors[1] = D_zero;
}

// Parameters
parameters {
  real<lower=0> length_scale;
  vector<lower=0>[r] signal_sd;
  vector<lower=0>[r] noise_sd;
  matrix[num_inducing, r] inducing_standard_normal;
}

// Transformed parameters
transformed parameters {
  matrix[num_inducing, r] inducing_values;

  // Draw the inducing values from their Gaussian-process prior
  for (j in 1:r) {
    // Build the inducing-point covariance matrix
    matrix[num_inducing, num_inducing] inducing_covariance =
      gp_exp_quad_cov(Z_vectors, signal_sd[j], length_scale);

    // Add numerical jitter for the Cholesky factorization
    inducing_covariance = add_diag(inducing_covariance, square(signal_sd[j]) * 1e-6);

    // Transform independent standard normals into inducing values
    inducing_values[, j] = cholesky_decompose(inducing_covariance) * inducing_standard_normal[, j];
  }
}

// Model
model {
  // Set the hyperparameter priors
  length_scale ~ inv_gamma(length_scale_prior_shape, length_scale_prior_scale);
  signal_sd ~ normal(0, signal_sd_prior_scale);
  noise_sd ~ normal(0, noise_sd_prior_scale);
  to_vector(inducing_standard_normal) ~ std_normal();

  // Fit a sparse Gaussian process for each target coordinate
  for (j in 1:r) {
    // Build the inducing-point covariance matrix
    matrix[num_inducing, num_inducing] inducing_covariance = gp_exp_quad_cov(Z_vectors, signal_sd[j], length_scale);

    // Add numerical jitter for the linear solves
    inducing_covariance = add_diag(inducing_covariance, square(signal_sd[j]) * 1e-6);

    // Build the pilot-to-inducing cross-covariance matrix
    matrix[S, num_inducing] cross_covariance = gp_exp_quad_cov(D_vectors, Z_vectors, signal_sd[j], length_scale);

    // Compute the inducing-point interpolation weights
    vector[num_inducing] inducing_weights = mdivide_left_spd(inducing_covariance, inducing_values[, j]);

    // Solve the inducing-to-pilot cross-covariance
    matrix[num_inducing, S] solved_cross_covariance = mdivide_left_spd(inducing_covariance, cross_covariance');

    // Compute the conditional Gaussian-process mean at each pilot discrepancy
    for (s in 1:S) {
      real f_mean = dot_product(cross_covariance[s]', inducing_weights);
      real f_variance = square(signal_sd[j]) - dot_product(cross_covariance[s]', solved_cross_covariance[, s]);

      // Add observation noise and model the pilot target
      C[s, j] ~ normal(f_mean, sqrt(square(noise_sd[j]) + fmax(f_variance, 0)));
    }
  }
}

// Generated quantities
generated quantities {
  vector[r] f_zero;

  // Draw the latent function at zero discrepancy for each target coordinate
  for (j in 1:r) {
    // Build the inducing-point covariance matrix
    matrix[num_inducing, num_inducing] inducing_covariance = gp_exp_quad_cov(Z_vectors, signal_sd[j], length_scale);

    // Add numerical jitter for the linear solves
    inducing_covariance = add_diag(inducing_covariance, square(signal_sd[j]) * 1e-6);

    // Build the inducing-to-zero cross-covariance matrix
    matrix[num_inducing, 1] zero_cross_covariance = gp_exp_quad_cov(Z_vectors, D_zero_vectors, signal_sd[j], length_scale);

    // Compute the conditional Gaussian-process mean at zero discrepancy
    real f_zero_mean = dot_product(zero_cross_covariance[, 1], mdivide_left_spd(inducing_covariance, inducing_values[, j]));

    // Compute the conditional Gaussian-process variance at zero discrepancy
    real f_zero_variance = square(signal_sd[j]) - dot_product(zero_cross_covariance[, 1], mdivide_left_spd(inducing_covariance, zero_cross_covariance[, 1]));

    // Draw the latent function at zero discrepancy
    f_zero[j] = normal_rng(f_zero_mean, sqrt(fmax(f_zero_variance, 0)));
  }
}

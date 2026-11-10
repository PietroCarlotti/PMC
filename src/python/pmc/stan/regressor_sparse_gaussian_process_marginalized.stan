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
}

// Model
model {
  // Set the hyperparameter priors
  length_scale ~ inv_gamma(length_scale_prior_shape, length_scale_prior_scale);
  signal_sd ~ normal(0, signal_sd_prior_scale);
  noise_sd ~ normal(0, noise_sd_prior_scale);

  // Fit a sparse Gaussian process for each target coordinate
  for (j in 1:r) {
    // Factor the inducing-point covariance, including numerical jitter
    matrix[num_inducing, num_inducing] inducing_covariance = gp_exp_quad_cov(Z_vectors, signal_sd[j], length_scale);
    inducing_covariance = add_diag(inducing_covariance, square(signal_sd[j]) * 1e-6);
    matrix[num_inducing, num_inducing] inducing_cholesky = cholesky_decompose(inducing_covariance);

    // Express each pilot's dependence on whitened inducing values
    matrix[S, num_inducing] cross_covariance = gp_exp_quad_cov(D_vectors, Z_vectors, signal_sd[j], length_scale);
    matrix[S, num_inducing] interpolation_weights = mdivide_left_tri_low(inducing_cholesky, cross_covariance')';

    // Observation noise plus the diagonal conditional GP variance
    vector[S] conditional_variance = square(noise_sd[j]) + fmax(rep_vector(square(signal_sd[j]), S) - rows_dot_product(interpolation_weights, interpolation_weights), rep_vector(0, S));

    // Posterior precision and mean of the whitened inducing values
    matrix[S, num_inducing] scaled_interpolation_weights = diag_pre_multiply(1 ./ sqrt(conditional_variance), interpolation_weights);
    matrix[num_inducing, num_inducing] posterior_precision = diag_matrix(rep_vector(1.0, num_inducing)) + scaled_interpolation_weights' * scaled_interpolation_weights;
    matrix[num_inducing, num_inducing] posterior_precision_cholesky = cholesky_decompose(posterior_precision);
    vector[num_inducing] posterior_information = interpolation_weights' * (C[, j] ./ conditional_variance);
    vector[num_inducing] posterior_inducing_mean = mdivide_right_tri_low(mdivide_left_tri_low(posterior_precision_cholesky, posterior_information)', posterior_precision_cholesky)';

    // Marginal Gaussian likelihood after integrating out the inducing values
    vector[S] residual = C[, j] - interpolation_weights * posterior_inducing_mean;
    real quadratic_form = dot_self(residual ./ sqrt(conditional_variance)) + dot_self(posterior_inducing_mean);

    target += -0.5 * (S * log(2 * pi()) + sum(log(conditional_variance)) + 2 * sum(log(diagonal(posterior_precision_cholesky))) + quadratic_form);
  }
}

// Generated quantities
generated quantities {
  vector[r] f_zero;

  // Draw the latent function at zero discrepancy for each target coordinate
  for (j in 1:r) {
    // Reconstruct the inducing posterior for this hyperparameter draw
    matrix[num_inducing, num_inducing] inducing_covariance = gp_exp_quad_cov(Z_vectors, signal_sd[j], length_scale);
    inducing_covariance = add_diag(inducing_covariance, square(signal_sd[j]) * 1e-6);
    matrix[num_inducing, num_inducing] inducing_cholesky = cholesky_decompose(inducing_covariance);

    matrix[S, num_inducing] cross_covariance = gp_exp_quad_cov(D_vectors, Z_vectors, signal_sd[j], length_scale);
    matrix[S, num_inducing] interpolation_weights = mdivide_left_tri_low(inducing_cholesky, cross_covariance')';
    vector[S] conditional_variance = square(noise_sd[j]) + fmax(rep_vector(square(signal_sd[j]), S) - rows_dot_product(interpolation_weights, interpolation_weights), rep_vector(0, S));

    matrix[S, num_inducing] scaled_interpolation_weights = diag_pre_multiply(1 ./ sqrt(conditional_variance), interpolation_weights);
    matrix[num_inducing, num_inducing] posterior_precision = diag_matrix(rep_vector(1.0, num_inducing)) + scaled_interpolation_weights' * scaled_interpolation_weights;
    matrix[num_inducing, num_inducing] posterior_precision_cholesky = cholesky_decompose(posterior_precision);
    vector[num_inducing] posterior_information = interpolation_weights' * (C[, j] ./ conditional_variance);
    vector[num_inducing] posterior_inducing_mean = mdivide_right_tri_low(mdivide_left_tri_low(posterior_precision_cholesky, posterior_information)', posterior_precision_cholesky)';

    // Combine conditional GP uncertainty with uncertainty in inducing values
    matrix[num_inducing, 1] zero_cross_covariance = gp_exp_quad_cov(Z_vectors, D_zero_vectors, signal_sd[j], length_scale);
    vector[num_inducing] zero_weights = mdivide_left_tri_low(inducing_cholesky, zero_cross_covariance[, 1]);
    real f_zero_mean = dot_product(zero_weights, posterior_inducing_mean);
    real f_zero_variance = fmax(square(signal_sd[j]) - dot_self(zero_weights) + dot_self(mdivide_left_tri_low(posterior_precision_cholesky, zero_weights)), 0);
    f_zero[j] = normal_rng(f_zero_mean, sqrt(f_zero_variance));
  }
}

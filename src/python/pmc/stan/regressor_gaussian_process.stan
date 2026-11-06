
// Data
data {
  int<lower=1> S;
  int<lower=1> m;
  int<lower=1> r;
  matrix[S, m] d;
  matrix[S, r] c;
  vector[m] d_zero;
}

// Transformed data
transformed data {
  array[S] vector[m] x;
  array[1] vector[m] x_zero;

  for (s in 1:S) {
    x[s] = d[s]';
  }
  x_zero[1] = d_zero;
}

// Parameters
parameters {
  real<lower=0> length_scale;
  vector<lower=0>[r] signal_sd;
  vector<lower=0>[r] noise_sd;
}

// Model
model {
  length_scale ~ inv_gamma(5, 5);
  signal_sd ~ std_normal();
  noise_sd ~ std_normal();

  for (j in 1:r) {
    matrix[S, S] K = gp_exp_quad_cov(x, signal_sd[j], length_scale);
    K = add_diag(K, square(noise_sd[j]));
    c[, j] ~ multi_normal_cholesky(
      rep_vector(0, S),
      cholesky_decompose(K)
    );
  }
}

// Generated quantities
generated quantities {
  vector[r] c_zero;

  for (j in 1:r) {
    matrix[S, S] K = gp_exp_quad_cov(x, signal_sd[j], length_scale);
    matrix[S, 1] k = gp_exp_quad_cov(
      x,
      x_zero,
      signal_sd[j],
      length_scale
    );
    K = add_diag(K, square(noise_sd[j]));
    c_zero[j] = dot_product(
      k[, 1],
      mdivide_left_spd(K, c[, j])
    );
  }
}

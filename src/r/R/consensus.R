#' Compute the arithmetic-mean consensus
#'
#' @param samples A numeric array with dimensions c(S, K, p)
#'
#' @export
#'
#' @return An S by p numeric matrix
consensus_mean <- function(samples) {
  # Validate samples
  dimensions <- validate_samples(samples)

  # Average over components
  consensus <- apply(samples, c(1L, 3L), mean)

  # Return an S by p matrix
  matrix(consensus, nrow = dimensions[[1L]], ncol = dimensions[[3L]])
}

#' Compute a weighted-average consensus
#'
#' @param samples A numeric array with dimensions c(S, K, p)
#' @param weights A numeric array with dimensions c(K, p, p)
#'
#' @export
#'
#' @return An S by p numeric matrix
consensus_weighted_average <- function(samples, weights) {
  # Check the weight shape
  if (!is.array(weights) || length(dim(weights)) != 3L) {
    stop("`weights` must be a numeric array with dimensions c(K, p, p)",
         call. = FALSE)
  }

  # Require numeric weights
  if (!is.numeric(weights)) {
    stop("`weights` must be numeric", call. = FALSE)
  }

  # Unpack weight dimensions
  weight_dimensions <- dim(weights)
  number_of_components <- weight_dimensions[[1L]]
  number_of_coordinates <- weight_dimensions[[2L]]

  # Check the weight dimensions
  if (number_of_components < 2L || number_of_coordinates < 1L ||
      weight_dimensions[[3L]] != number_of_coordinates) {
    stop("`weights` must have dimensions c(K, p, p) with K >= 2 and p >= 1",
         call. = FALSE)
  }

  # Check that weights sum to the identity
  weight_sum <- matrix(
    apply(weights, c(2L, 3L), sum),
    nrow = number_of_coordinates,
    ncol = number_of_coordinates
  )
  if (!isTRUE(all.equal(weight_sum, diag(number_of_coordinates),
                        tolerance = 1e-5))) {
    stop("`weights` must sum to the identity matrix over K", call. = FALSE)
  }

  # Validate samples
  sample_dimensions <- validate_samples(samples)
  number_of_draws <- sample_dimensions[[1L]]

  # Check that weights match the sample dimensions
  if (sample_dimensions[[2L]] != number_of_components ||
      sample_dimensions[[3L]] != number_of_coordinates) {
    stop("`weights` must match the K and p dimensions of `samples`",
         call. = FALSE)
  }

  # Apply each weight matrix and sum over components
  consensus <- matrix(0, nrow = number_of_draws,
                      ncol = number_of_coordinates)
  for (component in seq_len(number_of_components)) {
    component_samples <- matrix(
      samples[, component, , drop = FALSE],
      nrow = number_of_draws,
      ncol = number_of_coordinates
    )
    component_weights <- matrix(
      weights[component, , , drop = FALSE],
      nrow = number_of_coordinates,
      ncol = number_of_coordinates
    )
    consensus <- consensus + component_samples %*% t(component_weights)
  }

  consensus
}

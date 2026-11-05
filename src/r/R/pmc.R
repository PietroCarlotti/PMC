#' Estimate an expectation with Product Monte Carlo
#'
#' @param samples A numeric array with dimensions c(S, K, p)
#' @param consensus A function mapping samples to an S by p matrix
#' @param discrepancy A function mapping samples to an S by m matrix
#' @param h A function mapping consensus values to an S by r matrix
#' @param regressor A function returning an r-vector at zero discrepancy
#'
#' @export
#'
#' @return A numeric vector of length r
pmc <- function(
    samples,
    consensus = consensus_mean,
    discrepancy = discrepancy_anchored_differences,
    h = transform_identity,
    regressor = regressor_ols) {
  # Check component functions
  if (!is.function(consensus) || !is.function(discrepancy) ||
      !is.function(h) || !is.function(regressor)) {
    stop("`consensus`, `discrepancy`, `h`, and `regressor` must be functions",
         call. = FALSE)
  }

  # Validate samples
  sample_dimensions <- validate_samples(samples)
  number_of_draws <- sample_dimensions[[1L]]
  number_of_coordinates <- sample_dimensions[[3L]]

  # Compute the consensus
  consensus_values <- consensus(samples)
  consensus_dimensions <- validate_consensus(consensus_values)
  if (consensus_dimensions[[1L]] != number_of_draws ||
      consensus_dimensions[[2L]] != number_of_coordinates) {
    stop("`consensus` must return an S by p matrix", call. = FALSE)
  }

  # Compute the discrepancy
  discrepancy_values <- discrepancy(samples)
  if (!is.matrix(discrepancy_values) || !is.numeric(discrepancy_values) ||
      nrow(discrepancy_values) != number_of_draws ||
      ncol(discrepancy_values) < 1L) {
    stop("`discrepancy` must return an S by m numeric matrix with m >= 1",
         call. = FALSE)
  }

  # Apply the transform
  target <- h(consensus_values)
  if (!is.matrix(target) || !is.numeric(target) ||
      nrow(target) != number_of_draws || ncol(target) < 1L) {
    stop("`h` must return an S by r numeric matrix with r >= 1",
         call. = FALSE)
  }

  # Predict at zero discrepancy
  estimate <- regressor(discrepancy_values, target)
  if (!is.numeric(estimate) || !is.null(dim(estimate)) ||
      length(estimate) != ncol(target) || any(!is.finite(estimate))) {
    stop("`regressor` must return a finite numeric vector of length r",
         call. = FALSE)
  }

  estimate
}

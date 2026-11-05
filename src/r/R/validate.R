# Validate samples
#
# Samples are a numeric array with dimensions c(S, K, p)
validate_samples <- function(samples) {
  # Check the input shape
  if (!is.array(samples) || length(dim(samples)) != 3L) {
    stop("`samples` must be a numeric array with dimensions c(S, K, p).",
         call. = FALSE)
  }

  # Require numeric samples
  if (!is.numeric(samples)) {
    stop("`samples` must be numeric.", call. = FALSE)
  }

  # Unpack the number of draws, components, and coordinates
  dimensions <- dim(samples)

  # Check the minimum dimensions
  if (dimensions[[1L]] < 1L || dimensions[[2L]] < 2L ||
      dimensions[[3L]] < 1L) {
    stop("`samples` must have S >= 1, K >= 2, and p >= 1.", call. = FALSE)
  }

  # Return the validated dimensions
  invisible(dimensions)
}

# Validate consensus values
#
# Consensus values are a numeric matrix with dimensions c(S, p)
validate_consensus <- function(consensus) {
  # Check the input shape
  if (!is.matrix(consensus)) {
    stop("`consensus` must be a numeric matrix with dimensions c(S, p)",
         call. = FALSE)
  }

  # Require numeric consensus values
  if (!is.numeric(consensus)) {
    stop("`consensus` must be numeric", call. = FALSE)
  }

  # Unpack dimensions
  dimensions <- dim(consensus)

  # Check the minimum dimensions
  if (dimensions[[1L]] < 1L || dimensions[[2L]] < 1L) {
    stop("`consensus` must have S >= 1 and p >= 1", call. = FALSE)
  }

  # Return the validated dimensions
  invisible(dimensions)
}

# Validate regression data
validate_regression_data <- function(discrepancy, target) {
  # Check the discrepancy shape
  if (!is.matrix(discrepancy) || !is.numeric(discrepancy)) {
    stop("`discrepancy` must be a numeric matrix with dimensions c(S, m)",
         call. = FALSE)
  }

  # Check the target shape
  if (!is.matrix(target) || !is.numeric(target)) {
    stop("`target` must be a numeric matrix with dimensions c(S, r)",
         call. = FALSE)
  }

  # Unpack dimensions
  discrepancy_dimensions <- dim(discrepancy)
  target_dimensions <- dim(target)
  number_of_draws <- discrepancy_dimensions[[1L]]
  number_of_discrepancies <- discrepancy_dimensions[[2L]]
  number_of_targets <- target_dimensions[[2L]]

  # Check matching and minimum dimensions
  if (target_dimensions[[1L]] != number_of_draws || number_of_draws < 1L ||
      number_of_discrepancies < 1L || number_of_targets < 1L) {
    stop("`discrepancy` and `target` must have matching S >= 1, m >= 1, and r >= 1",
         call. = FALSE)
  }

  invisible(c(number_of_draws, number_of_discrepancies, number_of_targets))
}

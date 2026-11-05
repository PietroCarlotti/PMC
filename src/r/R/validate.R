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

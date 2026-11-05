#' Compute anchored differences
#'
#' @param samples A numeric array with dimensions c(S, K, p)
#'
#' @export
#'
#' @return An S by (K - 1) * p numeric matrix
discrepancy_anchored_differences <- function(samples) {
  # Validate samples
  dimensions <- validate_samples(samples)

  # Unpack dimensions
  number_of_draws <- dimensions[[1L]]
  number_of_components <- dimensions[[2L]]
  number_of_coordinates <- dimensions[[3L]]

  # Subtract the first component from the others
  anchor <- matrix(
    samples[, 1L, , drop = FALSE],
    nrow = number_of_draws,
    ncol = number_of_coordinates
  )
  differences <- array(
    0,
    dim = c(number_of_draws, number_of_components - 1L,
            number_of_coordinates)
  )
  for (component in 2L:number_of_components) {
    component_samples <- matrix(
      samples[, component, , drop = FALSE],
      nrow = number_of_draws,
      ncol = number_of_coordinates
    )
    differences[, component - 1L, ] <- component_samples - anchor
  }

  # Flatten the differences with component order before coordinate order
  matrix(
    aperm(differences, c(1L, 3L, 2L)),
    nrow = number_of_draws,
    ncol = (number_of_components - 1L) * number_of_coordinates
  )
}

#' Compute a norm discrepancy
#'
#' @param samples A numeric array with dimensions c(S, K, p)
#' @param order The norm order, at least 1 or Inf
#'
#' @export
#'
#' @return An S by 1 numeric matrix
discrepancy_norm <- function(samples, order = 2) {
  # Check the norm order
  if (!is.numeric(order) || length(order) != 1L || is.na(order) ||
      order < 1) {
    stop("`order` must be at least 1 or Inf", call. = FALSE)
  }

  # Compute anchored differences
  differences <- discrepancy_anchored_differences(samples)

  # Compute the norm of each difference vector
  if (is.infinite(order)) {
    discrepancy <- apply(abs(differences), 1L, max)
  } else {
    discrepancy <- rowSums(abs(differences)^order)^(1 / order)
  }

  matrix(discrepancy, ncol = 1L)
}

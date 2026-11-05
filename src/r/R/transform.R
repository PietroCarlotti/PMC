#' Apply the identity transform
#'
#' @param consensus An S by p numeric matrix
#'
#' @export
#'
#' @return An S by p numeric matrix
transform_identity <- function(consensus) {
  # Validate consensus values
  validate_consensus(consensus)

  # Return consensus unchanged
  consensus
}

#' Apply the elementwise-square transform
#'
#' @param consensus An S by p numeric matrix
#'
#' @export
#'
#' @return An S by p numeric matrix
transform_hadamard_product <- function(consensus) {
  # Validate consensus values
  validate_consensus(consensus)

  # Square each coordinate
  consensus^2
}

#' Apply the inner-product transform
#'
#' @param consensus An S by p numeric matrix
#'
#' @export
#'
#' @return An S by 1 numeric matrix
transform_inner_product <- function(consensus) {
  # Validate consensus values
  validate_consensus(consensus)

  # Sum squared coordinates
  matrix(rowSums(consensus^2), ncol = 1L)
}

#' Apply the outer-product transform
#'
#' @param consensus An S by p numeric matrix
#'
#' @export
#'
#' @return An S by p squared numeric matrix
transform_outer_product <- function(consensus) {
  # Validate consensus values
  dimensions <- validate_consensus(consensus)
  number_of_draws <- dimensions[[1L]]
  number_of_coordinates <- dimensions[[2L]]

  # Compute and flatten each outer product
  outer_products <- matrix(
    0,
    nrow = number_of_draws,
    ncol = number_of_coordinates^2
  )
  for (draw in seq_len(number_of_draws)) {
    outer_products[draw, ] <- as.vector(t(tcrossprod(consensus[draw, ])))
  }

  outer_products
}

#' Apply a coordinatewise threshold indicator
#'
#' @param consensus An S by p numeric matrix
#' @param thresholds A numeric vector of length p
#'
#' @export
#'
#' @return An S by p numeric matrix
transform_indicator_threshold <- function(consensus, thresholds) {
  # Check thresholds
  if (!is.numeric(thresholds) || !is.null(dim(thresholds)) ||
      length(thresholds) < 1L || any(!is.finite(thresholds))) {
    stop("`thresholds` must be a nonempty finite numeric vector",
         call. = FALSE)
  }

  # Validate consensus values
  dimensions <- validate_consensus(consensus)
  number_of_draws <- dimensions[[1L]]
  number_of_coordinates <- dimensions[[2L]]

  # Check the number of thresholds
  if (length(thresholds) != number_of_coordinates) {
    stop("`thresholds` must have one value per consensus coordinate",
         call. = FALSE)
  }

  # Return coordinatewise indicators
  threshold_matrix <- matrix(
    thresholds,
    nrow = number_of_draws,
    ncol = number_of_coordinates,
    byrow = TRUE
  )
  1 * (consensus > threshold_matrix)
}

#' Apply a Euclidean-ball indicator
#'
#' @param consensus An S by p numeric matrix
#' @param center A numeric vector of length p
#' @param radius A positive numeric scalar
#'
#' @export
#'
#' @return An S by 1 numeric matrix
transform_indicator_ball <- function(consensus, center, radius) {
  # Check the center
  if (!is.numeric(center) || !is.null(dim(center)) || length(center) < 1L ||
      any(!is.finite(center))) {
    stop("`center` must be a nonempty finite numeric vector", call. = FALSE)
  }

  # Check the radius
  if (!is.numeric(radius) || length(radius) != 1L || !is.finite(radius) ||
      radius <= 0) {
    stop("`radius` must be finite and positive", call. = FALSE)
  }

  # Validate consensus values
  dimensions <- validate_consensus(consensus)
  number_of_draws <- dimensions[[1L]]
  number_of_coordinates <- dimensions[[2L]]

  # Check the number of center coordinates
  if (length(center) != number_of_coordinates) {
    stop("`center` must have one value per consensus coordinate",
         call. = FALSE)
  }

  # Return indicators for points inside the ball
  center_matrix <- matrix(
    center,
    nrow = number_of_draws,
    ncol = number_of_coordinates,
    byrow = TRUE
  )
  distance <- sqrt(rowSums((consensus - center_matrix)^2))
  matrix(1 * (distance <= radius), ncol = 1L)
}

#' Apply a coordinatewise box indicator
#'
#' @param consensus An S by p numeric matrix
#' @param lower A numeric vector of lower bounds of length p
#' @param upper A numeric vector of upper bounds of length p
#'
#' @export
#'
#' @return An S by p numeric matrix
transform_indicator_box <- function(consensus, lower, upper) {
  # Check the bounds
  if (!is.numeric(lower) || !is.numeric(upper) || !is.null(dim(lower)) ||
      !is.null(dim(upper)) || length(lower) < 1L ||
      length(lower) != length(upper) || any(!is.finite(lower)) ||
      any(!is.finite(upper))) {
    stop("`lower` and `upper` must be nonempty finite numeric vectors of equal length",
         call. = FALSE)
  }

  # Check the bound order
  if (any(lower > upper)) {
    stop("`lower` must not exceed `upper` in any coordinate", call. = FALSE)
  }

  # Validate consensus values
  dimensions <- validate_consensus(consensus)
  number_of_draws <- dimensions[[1L]]
  number_of_coordinates <- dimensions[[2L]]

  # Check the number of bounds
  if (length(lower) != number_of_coordinates) {
    stop("`lower` and `upper` must have one value per consensus coordinate",
         call. = FALSE)
  }

  # Return coordinatewise indicators
  lower_matrix <- matrix(
    lower,
    nrow = number_of_draws,
    ncol = number_of_coordinates,
    byrow = TRUE
  )
  upper_matrix <- matrix(
    upper,
    nrow = number_of_draws,
    ncol = number_of_coordinates,
    byrow = TRUE
  )
  1 * ((consensus >= lower_matrix) & (consensus <= upper_matrix))
}

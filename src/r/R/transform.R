#' Apply the identity transform
#'
#' @param consensus An S by p numeric matrix
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
#' @return An S by p numeric matrix
transform_hadamard_product <- function(consensus) {
  # Validate consensus values
  validate_consensus(consensus)

  # Square each coordinate
  consensus^2
}

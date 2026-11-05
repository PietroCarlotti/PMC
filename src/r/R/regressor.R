# Compute a matrix pseudoinverse
matrix_pseudoinverse <- function(x, tolerance) {
  # Compute the singular value decomposition
  decomposition <- svd(x)
  singular_values <- decomposition$d

  # Return zeros when every singular value is negligible
  if (length(singular_values) == 0L || max(singular_values) == 0) {
    return(matrix(0, nrow = ncol(x), ncol = nrow(x)))
  }

  # Keep singular values above the tolerance
  keep <- singular_values > tolerance * max(singular_values)
  if (!any(keep)) {
    return(matrix(0, nrow = ncol(x), ncol = nrow(x)))
  }

  decomposition$v[, keep, drop = FALSE] %*%
    (t(decomposition$u[, keep, drop = FALSE]) / singular_values[keep])
}

#' Fit ordinary least squares at zero discrepancy
#'
#' @param discrepancy An S by m numeric matrix
#' @param target An S by r numeric matrix
#'
#' @export
#'
#' @return A numeric vector of length r
regressor_ols <- function(discrepancy, target) {
  # Validate regression data
  dimensions <- validate_regression_data(discrepancy, target)
  number_of_draws <- dimensions[[1L]]
  number_of_discrepancies <- dimensions[[2L]]

  # Use centered normal equations with ample draws
  if (number_of_draws >= 2L * (number_of_discrepancies + 1L)) {
    discrepancy_mean <- colMeans(discrepancy)
    target_mean <- colMeans(target)
    gram <- crossprod(discrepancy) -
      number_of_draws * tcrossprod(discrepancy_mean)
    cross <- crossprod(discrepancy, target) -
      number_of_draws * tcrossprod(discrepancy_mean, target_mean)
    coefficients <- solve(gram, cross)
    intercept <- target_mean -
      as.vector(discrepancy_mean %*% coefficients)
  } else if (number_of_draws >= number_of_discrepancies + 1L) {
    # Solve least squares with an intercept
    design <- cbind(1, discrepancy)
    coefficients <- matrix_pseudoinverse(design, tolerance = 1e-5) %*% target
    intercept <- coefficients[1L, ]
  } else {
    # Fit minimum-norm slopes and recover the intercept
    discrepancy_mean <- colMeans(discrepancy)
    target_mean <- colMeans(target)
    centered_discrepancy <- sweep(discrepancy, 2L, discrepancy_mean)
    centered_target <- sweep(target, 2L, target_mean)
    coefficients <- matrix_pseudoinverse(
      centered_discrepancy,
      tolerance = .Machine$double.eps * max(dim(centered_discrepancy))
    ) %*% centered_target
    intercept <- target_mean -
      as.vector(discrepancy_mean %*% coefficients)
  }

  # Require a finite prediction
  if (any(!is.finite(intercept))) {
    stop("OLS must return finite values", call. = FALSE)
  }

  as.vector(intercept)
}
